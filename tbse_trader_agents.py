"""Module containing all trader algos"""
# pylint: disable=too-many-lines
import math
import random
import sys
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from tbse_msg_classes import Order
from tbse_sys_consts import TBSE_SYS_MAX_PRICE, TBSE_SYS_MIN_PRICE

# pylint: disable=too-many-instance-attributes
class Trader:
    """Trader superclass - mostly unchanged from original BSE code by Dave Cliff
    all Traders have a trader id, bank balance, blotter, and list of orders to execute"""

    def __init__(self, ttype, tid, balance, time):
        self.ttype = ttype  # what type / strategy this trader is
        self.tid = tid  # trader unique ID code
        self.balance = balance  # money in the bank
        self.blotter = []  # record of trades executed
        self.orders = {}  # customer orders currently being worked (fixed at 1)
        self.n_quotes = 0  # number of quotes live on LOB
        self.willing = 1  # used in ZIP etc
        self.able = 1  # used in ZIP etc
        self.birth_time = time  # used when calculating age of a trader/strategy
        self.profit_per_time = 0  # profit per unit t
        self.n_trades = 0  # how many trades has this trader done?
        self.last_quote = None  # record of what its last quote was
        self.times = [0, 0, 0, 0]  # values used to calculate timing elements

    def log(self, message):
        self.logger.debug(f"{self.ttype}_{self.tid} at time {self.time}: {message}")

    def __str__(self):
        return f'[TID {self.tid} type {self.ttype} balance {self.balance} blotter {self.blotter} ' \
               f'orders {self.orders} n_trades {self.n_trades} profit_per_time {self.profit_per_time}]'

    def add_order(self, order, verbose):
        """
        Adds an order to the traders list of orders
        in this version, trader has at most one order,
        if allow more than one, this needs to be self.orders.append(order)
        :param order: the order to be added
        :param verbose: should verbose logging be printed to console
        :return: Response: "Proceed" if no current offer on LOB, "LOB_Cancel" if there is an order on the LOB needing
                 cancelled.\
        """

        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders[order.coid] = order

        if verbose:
            print(f'add_order < response={response}')
        return response

    def del_order(self, coid):
        """
        Removes current order from traders list of orders
        :param coid: Customer order ID of order to be deleted
        """
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        # CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
        self.orders.pop(coid)

    def bookkeep(self, trade, order, verbose, time):
        """
        Updates trader's internal stats with trade and order
        :param trade: Trade that has been executed
        :param order: Order trade was in response to
        :param verbose: Should verbose logging be printed to console
        :param time: Current time
        """
        output_string = ""

        if trade['coid'] in self.orders:
            coid = trade['coid']
            order_price = self.orders[coid].price
        elif trade['counter'] in self.orders:
            coid = trade['counter']
            order_price = self.orders[coid].price
        else:
            print("COID not found")
            sys.exit("This is non ideal ngl.")

        self.blotter.append(trade)  # add trade record to trader's blotter
        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transaction_price = trade['price']
        if self.orders[coid].otype == 'Bid':
            profit = order_price - transaction_price
        else:
            profit = transaction_price - order_price
        self.balance += profit
        self.n_trades += 1
        self.profit_per_time = self.balance / (time - self.birth_time)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            print(str(trade['coid']) + " " + str(trade['counter']) + " " + str(order.coid) + " " + str(
                self.orders[0].coid))
            sys.exit()

        if verbose:
            print(f'{output_string} profit={profit} balance={self.balance} profit/t={self.profit_per_time}')
        self.del_order(coid)  # delete the order

    # pylint: disable=unused-argument,no-self-use
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        specify how trader responds to events in the market
        this is a null action, expect it to be overloaded by specific algos
        :param time: Current time
        :param lob: Limit order book
        :param trade: Trade being responded to
        :param verbose: Should verbose logging be printed to console
        :return: Unused
        """
        return None

    # pylint: disable=unused-argument,no-self-use
   
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Get's the traders order based on the current state of the market
        :param time: Current time
        :param countdown: Time to end of session
        :param lob: Limit order book
        :return: The order
        """
        return None


class TraderGiveaway(Trader):
    """
    Trader subclass Giveaway
    even dumber than a ZI-U: just give the deal away
    (but never makes a loss)
    """
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Get's giveaway traders order - in this case the price is just the limit price from the customer order
        :param time: Current time
        :param countdown: Time until end of session
        :param lob: Limit order book
        :return: Order to be sent to the exchange
        """

        if len(self.orders) < 1:
            order = None
        else:
            coid = max(self.orders.keys())     # Find the price limit
            quote_price = self.orders[coid].price
            order = Order(self.tid,
                          self.orders[coid].otype,
                          quote_price,
                          self.orders[coid].qty,
                          time, self.orders[coid].coid, self.orders[coid].toid)
            self.last_quote = order
        return order


class TraderVWAP(Trader):
    def __init__(self, ttype, tid, balance, time):
        super().__init__(ttype, tid, balance, time)
        self.window_size = 10
        self.prices = []
        self.volumes = []
        self.current_batch_prices = []
        self.current_batch_volumes = []
        self.last_batch = None

    def calculate_vwap(self):
        if len(self.prices) < self.window_size:
            return None
        prices = np.array(self.prices[-self.window_size:])
        volumes = np.array(self.volumes[-self.window_size:])
        return np.sum(prices * volumes) / np.sum(volumes)

    def update_vwap(self, price, volume):
        self.prices.append(price)
        self.volumes.append(volume)
        if len(self.prices) > self.window_size:
            self.prices.pop(0)
            self.volumes.pop(0)

    def end_of_batch(self):
        if self.current_batch_prices:
            avg_price = np.mean(self.current_batch_prices)
            total_volume = np.sum(self.current_batch_volumes)
            self.update_vwap(avg_price, total_volume)
        self.current_batch_prices = []
        self.current_batch_volumes = []

    def respond(self, time, p_eq, q_eq, demand_curve, supply_curve, lob, trades, verbose):
        if self.last_batch == (demand_curve, supply_curve):
            return
        else:
            self.last_batch = (demand_curve, supply_curve)

        # Process trades
        for trade in trades:
            self.current_batch_prices.append(trade['price'])
            self.current_batch_volumes.append(trade['qty'])

        self.end_of_batch()

    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        if len(self.orders) < 1:
            self.active = False
            return None

        self.active = True
        coid = max(self.orders.keys())
        limit_price = self.orders[coid].price
        otype = self.orders[coid].otype

        vwap = self.calculate_vwap()
        if vwap is None:
            vwap = limit_price  # If we don't have enough data, use limit price

        if otype == 'Bid':
            quote_price = min(vwap, limit_price)
        else:  # otype == 'Ask'
            quote_price = max(vwap, limit_price)

        order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                      self.orders[coid].toid)
        self.last_quote = order
        return order

    def bookkeep(self, trade, order, verbose, time):
        super().bookkeep(trade, order, verbose, time)
        # Update VWAP after a successful trade
        if trade['party1'] == self.tid or trade['party2'] == self.tid:
            self.update_vwap(trade['price'], trade['qty'])


class TraderShaver(Trader):
    """Trader subclass Shaver
    shaves a penny off the best price
    if there is no best price, creates "stub quote" at system max/min"""
    def get_order(self, time, p_eq , q_eq, demand_curve, supply_curve, countdown, lob):
        """
        Get's Shaver trader order by shaving/adding a penny to current best bid
        :param time: Current time
        :param countdown: Countdown to end of market session
        :param lob: Limit order book
        :return: The trader order to be sent to the exchange
        """
        if len(self.orders) < 1:
            order = None
        else:

            coid = max(self.orders.keys())
            limit_price = self.orders[coid].price
            otype = self.orders[coid].otype

            best_bid = 500
            best_ask = 0

            if demand_curve!=[]:
                best_bid = max(demand_curve, key=lambda x: x[0])[0]+1

            if supply_curve!=[]:
                best_ask = min(supply_curve, key=lambda x: x[0])[0]-1

            if otype == 'Bid':
                quote_price= best_bid
                quote_price = min(quote_price, limit_price)
            else:
                quote_price = best_ask
                quote_price = max(quote_price, limit_price)

            #quote_price = min(quote_price, limit_price)
            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order
        return order


# Trader subclass ZIP
# After Cliff 1997
# pylint: disable=too-many-instance-attributes
class TraderZip(Trader):
    """ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
    NB this implementation keeps separate margin values for buying & selling,
       so a single trader can both buy AND sell
       -- in the original, traders were either buyers OR sellers"""

    def __init__(self, ttype, tid, balance, time):

        Trader.__init__(self, ttype, tid, balance, time)
        m_fix = 0.05
        m_var = 0.3
        self.job = None  # this is 'Bid' or 'Ask' depending on customer order
        self.active = False  # gets switched to True while actively working an order
        self.prev_change = 0  # this was called last_d in Cliff'97
        self.beta = 0.2 + 0.2 * random.random()  # learning rate #0.1 + 0.2 * random.random()
        self.momentum = 0.3 * random.random()  # momentum #0.3 * random.random()
        self.ca = 0.10  # self.ca & .cr were hard-coded in '97 but parameterised later
        self.cr = 0.10
        self.margin = None  # this was called profit in Cliff'97
        self.margin_buy = -1.0 * (m_fix + m_var * random.random())
        self.margin_sell = m_fix + m_var * random.random()
        self.price = None
        self.limit = None
        self.times = [0, 0, 0, 0]
        # memory of best price & quantity of best bid and ask, on LOB on previous update
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None
        self.last_batch = None

    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        :param time: Current time
        :param countdown: Time until end of current market session
        :param lob: Limit order book
        :return: Trader order to be sent to exchange
        """
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            coid = max(self.orders.keys())
            self.active = True
            self.limit = self.orders[coid].price
            self.job = self.orders[coid].otype
            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                self.margin = self.margin_buy
            else:
                # currently a seller (working a sell order)
                self.margin = self.margin_sell
            quote_price = int(self.limit * (1 + self.margin))
            self.price = quote_price

            order = Order(self.tid, self.job, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order
        return order

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        update margin on basis of what happened in marke
        ZIP trader responds to market events, altering its margin
        does this whether it currently has an order to work or not
        :param time: Current time
        :param lob: Limit order book
        :param trade: Trade being responded to
        :param verbose: Should verbose logging be printed to console
        """

        if self.last_batch==(demand_curve,supply_curve):
            return
        else:
            self.last_batch = (demand_curve,supply_curve)

        trade = trades[0] if trades else None

        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if demand_curve!=[]:
            best_bid = max(demand_curve, key=lambda x: x[0])[0]
        if supply_curve!=[]:
            best_ask = min(supply_curve, key=lambda x: x[0])[0]

        def target_up(price):
            """
            generate a higher target price by randomly perturbing given price
            :param price: Current price
            :return: New price target
            """
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel + ptrb_abs, 0))

            return target

        def target_down(price):
            """
            generate a lower target price by randomly perturbing given price
            :param price: Current price
            :return: New price target
            """
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel - ptrb_abs, 0))

            return target

        def willing_to_trade(price):
            """
            am I willing to trade at this price?
            :param price: Price to be traded out
            :return: Is the trader willing to trade
            """
            willing = False
            if self.job == 'Bid' and self.active and self.price >= price:
                willing = True
            if self.job == 'Ask' and self.active and self.price <= price:
                willing = True
            return willing

        def profit_alter(price):
            """
            Update target profit margin
            :param price: New target profit margin
            """
            old_price = self.price
            diff = price - old_price
            change = ((1.0 - self.momentum) * (self.beta * diff)) + (self.momentum * self.prev_change)
            self.prev_change = change
            new_margin = ((self.price + change) / self.limit) - 1.0

            if self.job == 'Bid':
                if new_margin < 0.0:
                    self.margin_buy = new_margin
                    self.margin = new_margin
            else:
                if new_margin > 0.0:
                    self.margin_sell = new_margin
                    self.margin = new_margin

            # set the price from limit and profit-margin
            self.price = int(round(self.limit * (1.0 + self.margin), 0))

        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False

        #lob_best_bid_p = lob['bids']['best']
        lob_best_bid_p = best_bid #CHANGE HERE
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = 1
            if self.prev_best_bid_p is None:
                self.prev_best_bid_p = lob_best_bid_p
            elif self.prev_best_bid_p < lob_best_bid_p:
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1] #might have to check if has been cancelled at some point during batch
            #for item in lob['tape'] check if cancel happened with price of
            if last_tape_item['type'] == 'Cancel':
                #print("Last bid was cancelled") #test.csv
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        #lob_best_ask_p = lob['asks']['best']
        lob_best_ask_p = best_ask #CHANGE HERE
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = 1
            if self.prev_best_ask_p is None:
                self.prev_best_ask_p = lob_best_ask_p
            elif self.prev_best_ask_p > lob_best_ask_p:
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                #print("Last bid was cancelled") # test.csv
                ask_lifted = False
            else:
                ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted

        if trade is None:
            deal = False

        if self.job == 'Ask':
            # seller
            if deal:
                trade_price = trade['price']
                if self.price <= trade_price:
                    # could sell for more? raise margin
                    target_price = target_up(trade_price)
                    profit_alter(target_price)
                elif ask_lifted and self.active and not willing_to_trade(trade_price):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_down(trade_price)
                    profit_alter(target_price)
            else:
                # no deal: aim for a target price higher than best bid
                if ask_improved and self.price > lob_best_ask_p:
                    if lob_best_bid_p is not None:
                        target_price = target_up(lob_best_bid_p)
                    else:
                        target_price = lob['asks']['worst']  # stub quote
                    profit_alter(target_price)

        if self.job == 'Bid':
            # buyer
            if deal:
                trade_price = trade['price']
                if self.price >= trade_price:
                    # could buy for less? raise margin (i.e. cut the price)
                    target_price = target_down(trade_price)
                    profit_alter(target_price)
                elif bid_hit and self.active and not willing_to_trade(trade_price):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_up(trade_price)
                    profit_alter(target_price)
            else:
                # no deal: aim for target price lower than best ask
                if bid_improved and self.price < lob_best_bid_p:
                    if lob_best_ask_p is not None:
                        target_price = target_down(lob_best_ask_p)
                    else:
                        target_price = lob['bids']['worst']  # stub quote
                    profit_alter(target_price)

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q


# pylint: disable=too-many-instance-attributes
class TraderIFB(Trader):
    def __init__(self, ttype, tid, balance, time):
        super().__init__(ttype, tid, balance, time)
        self.order_book_imbalance_threshold = 0.2
        self.gvwy_threshold = 0.1  # The threshold of using the gvwy algorithm
        self.last_batch = None
        self.performance_history = []  # The performance tracker
        self.previous_balance = balance
        self.last_strategy = None

    @staticmethod
    def analyze_order_book(demand_curve, supply_curve):
        if not demand_curve or not supply_curve:
            return 0
        bid_volume = sum(q for _, q in demand_curve)
        ask_volume = sum(q for _, q in supply_curve)
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0

    def decide_strategy(self, order_imbalance):
        if abs(order_imbalance) < self.gvwy_threshold:
            return 'GVWY'
        elif len(self.performance_history) > 10:
            gvwy_performance = sum(p for s, p in self.performance_history if s == 'GVWY')
            ifb_performance = sum(p for s, p in self.performance_history if s == 'IFB')
            return 'GVWY' if gvwy_performance > ifb_performance else 'IFB'
        else:
            return 'IFB'

    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        if len(self.orders) < 1:
            self.active = False
            return None

        if self.last_batch == (demand_curve, supply_curve):
            return None
        self.last_batch = (demand_curve, supply_curve)

        coid = max(self.orders.keys())
        self.active = True
        limit_price = self.orders[coid].price
        otype = self.orders[coid].otype

        order_imbalance = self.analyze_order_book(demand_curve, supply_curve)
        self.last_strategy = self.decide_strategy(order_imbalance)

        if self.last_strategy == 'GVWY':
            quote_price = limit_price
        else:
            if abs(order_imbalance) > self.order_book_imbalance_threshold:
                if order_imbalance > 0:  # Bullish signal
                    if otype == 'Bid':
                        quote_price = limit_price
                    else:
                        quote_price = int(limit_price * 1.05)
                else:  # Bearish signal
                    if otype == 'Bid':
                        quote_price = int(limit_price * 0.95)
                    else:
                        quote_price = limit_price
            else:
                quote_price = limit_price

        quote_price = max(TBSE_SYS_MIN_PRICE, min(quote_price, TBSE_SYS_MAX_PRICE))

        order = Order(self.tid, otype, quote_price, self.orders[coid].qty,
                      time, self.orders[coid].coid, self.orders[coid].toid)
        self.last_quote = order
        return order

    def bookkeep(self, trade, order, verbose, time):
        super().bookkeep(trade, order, verbose, time)
        profit = self.balance - self.previous_balance
        self.performance_history.append((self.last_strategy, profit))
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        self.previous_balance = self.balance

    def respond(self, time, p_eq, q_eq, demand_curve, supply_curve, lob, trades, verbose):
        # 如果需要在每个批次中响应，可以在这里添加逻辑
        pass


# pylint: disable=too-many-instance-attributes
class TraderGdx(Trader):
    def __init__(self, ttype, tid, balance, time):
        super().__init__(ttype, tid, balance, time)
        self.theta0 = 100  # threshold-function limit value
        self.m = 4  # tangent-function multiplier
        self.strat_range_min = -1.0
        self.strat_range_max = 1.0
        self.strat = random.uniform(self.strat_range_min, self.strat_range_max)
        self.lastquote = None
        self.job = None
        self.active = False

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            self.active = False
            return None

        self.active = True
        order = self.orders[0]
        self.job = order.otype
        quoteprice = self.calc_price(lob, order)
        new_order = Order(self.tid, self.job, quoteprice, order.qty, time, lob['QID'])
        self.lastquote = new_order
        return new_order

    def calc_price(self, lob, order):
        def get_price_range(order):
            if order.otype == 'Bid':
                best_ask = lob['asks']['best']
                return best_ask if best_ask is not None else order.price, order.price
            else:
                best_bid = lob['bids']['best']
                return order.price, best_bid if best_bid is not None else order.price

        def calc_cdf(price, min_price, max_price):
            if max_price == min_price:
                return 1.0 if price >= max_price else 0.0

            c = self.m * math.tan(math.pi * (self.strat + 0.5))
            c = max(-self.theta0, min(self.theta0, c))

            if abs(c) < 1e-6:
                c = 1e-6 if c >= 0 else -1e-6

            rel_price = (price - min_price) / (max_price - min_price)
            if order.otype == 'Bid':
                return (math.exp(c * rel_price) - 1) / (math.exp(c) - 1)
            else:
                return (math.exp(c * (1 - rel_price)) - 1) / (math.exp(c) - 1)

        min_price, max_price = get_price_range(order)

        u = random.random()
        for p in range(int(min_price), int(max_price) + 1):
            if u <= calc_cdf(p, min_price, max_price):
                return p

        # 如果没有找到合适的价格，返回限价
        return order.price

    # ----------------trader-types have all been defined now-------------
