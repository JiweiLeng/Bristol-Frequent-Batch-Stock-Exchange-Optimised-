"""TBSE Config File"""
# pylint: skip-file
# BFBSE 

batch_interval = 0.25  # interval between batches in number of seconds.

# General
sessionLength = 1  # Length of session in seconds.
virtualSessionLength = 600  # Number of virtual timesteps per sessionLength.
verbose = False  # Adds additional output for debugging. #changed this to True

# BSE ONLY
start_time = 0.0
end_time = 600.0

# Trader Schedule
# Define number of each algorithm used one side of exchange (buyers or sellers)
# Same values will be used to define other side of exchange (buyers = sellers)
# Input sequence follows the same
numZIC = 0
numZIP = 0
numRaForest = 5
numMIX = 5
numGVWY = 0
numAA = 0

# Noise Traders
numRandom = 5
numHerd = 5

# Order Schedule
useOffset = True  # Use an offset function to vary equilibrium price, this is disabled if useInputFile = True #causes
# multiple prints sometimes?
useInputFile = True  # Use an input file to define order schedule (e.g. Real World Trading data)
enable_market_shocks = True  # Master switch for all market shocks

input_file = "RWD/IBM-310817.csv"  # Path to real world data input file
stepmode = 'random'  # Valid values: 'fixed', 'jittered', 'random'
timemode = 'periodic'  # Valid values: 'periodic', 'drip-fixed', 'drip-jitter', 'drip-poisson'
interval = 30  # Virtual seconds between new set of customer orders being generated. #changed to 250 from 30

# Market supply schedule
supply = {
    'rangeMax': {  # Range of values between which the max possible sell order will be randomly placed
        'rangeHigh': 200,
        'rangeLow': 100
    }, 'rangeMin': {  # Range of values between which the min possible sell order will be randomly placed
        'rangeHigh': 100,
        'rangeLow': 0
    }
}

# NOTE: If symmetric = True this schedule is ignored and the demand schedule will equal the above supply schedule.
demand = {
    'rangeMax': {  # Range of values between which the max possible buy order will be randomly placed
        'rangeHigh': 200,
        'rangeLow': 100
    }, 'rangeMin': {  # Range of values between which the min possible buy order will be randomly placed
        'rangeHigh': 100,
        'rangeLow': 0
    }
}

# For single schedule: using config trader schedule, or command-line trader schedule.
numTrials = 50

# For multiple schedules: using input csv file. 
numSchedulesPerRatio = 10  # Number of schedules per ratio of traders in csv file.
numTrialsPerSchedule = 100  # Number of trails per schedule.
symmetric = True  # Should range of supply = range of demand?

# Micro-market shock configuration
micro_shock_probability = 0.01  # Trigger probability
micro_shock_order_count_min = 1  # Minimum number of orders added to
micro_shock_order_count_max = 5  # Maximum number of orders added to

# Macro-market shock configuration
macro_shock_probability = 0.0001  # Trigger probability
macro_shock_duration_min = 10  # Min Duration
macro_shock_duration_max = 30  # Max Duration
macro_shock_intensity_min = 0.01  # Min intensity
macro_shock_intensity_max = 0.05  # Max intensity

# Function for parsing config values.
def parse_config():
    valid = True
    if not isinstance(sessionLength, int):
        print("CONFIG ERROR: sessionLengths must be integer.")
        valid = False
    if not isinstance(virtualSessionLength, int):
        print("CONFIG ERROR: virtualSessionLengths must be integer.")
        valid = False
    if not isinstance(verbose, bool):
        print("CONFIG ERROR: verbose must be bool.")
        valid = False
    if not isinstance(start_time, float):
        print("CONFIG ERROR: start_time must be a float.")
        valid = False
    if not isinstance(end_time, float):
        print("CONFIG ERROR: end_time must be a float.")
        valid = False
    if not (isinstance(numZIC, int) and isinstance(numMIX, int) and isinstance(numRaForest, int) and
            isinstance(numGVWY, int) and isinstance(numAA, int) and isinstance(numZIP, int)):
        print("CONFIG ERROR: Trader schedule values must be integer.")
        valid = False
    if not isinstance(useOffset, bool):
        print("CONFIG ERROR: useOffset must be bool.")
        valid = False
    if not isinstance(stepmode, str):
        print("CONFIG ERROR: stepmode must be string.")
        valid = False
    if not isinstance(timemode, str):
        print("CONFIG ERROR: timemode must be string.")
        valid = False
    if not isinstance(interval, int):
        print("CONFIG ERROR: interval must be integer.")
        valid = False
    if not (isinstance(supply['rangeMax']['rangeHigh'], int) and isinstance(supply['rangeMax']['rangeLow'],
                                                                            int) and isinstance(
        supply['rangeMin']['rangeHigh'], int) and isinstance(supply['rangeMin']['rangeLow'], int) and
            isinstance(demand['rangeMax']['rangeHigh'], int) and isinstance(demand['rangeMax']['rangeLow'],
                                                                            int) and isinstance(
                demand['rangeMin']['rangeHigh'], int) and isinstance(demand['rangeMin']['rangeLow'], int)):
        print("CONFIG ERROR: Trader schedule values must be integer.")
        valid = False
    if not isinstance(numTrials, int):
        print("CONFIG ERROR: numTrials must be integer.")
        valid = False
    if not isinstance(numSchedulesPerRatio, int):
        print("CONFIG ERROR: numSchedulesPerRatio must be integer.")
        valid = False
    if not isinstance(numTrialsPerSchedule, int):
        print("CONFIG ERROR: numTrialsPerSchedule must be integer.")
        valid = False
    if not isinstance(symmetric, bool):
        print("CONFIG ERROR: symmetric must be bool.")
        valid = False

    if not valid:
        return False

    if sessionLength <= 0 or virtualSessionLength <= 0:
        print("CONFIG ERROR: Session lengths must be greater than 0.")
        valid = False
    if start_time < 0:
        print("CONFIG ERROR: start_time must be greater than or equal to 0.")
        valid = False
    if end_time <= start_time:
        print("CONFIG ERROR: end_time must be greater than start_time")
        valid = False
    if numMIX < 0 or numRaForest < 0 or numGVWY < 0 or numAA < 0 or numZIC < 0 or numZIP < 0:
        print("CONFIG ERROR: All trader schedule values must be greater than or equal to 0.")
        valid = False
    if stepmode not in ['fixed', 'jittered', 'random']:
        print("CONFIG ERROR: stepmode must be 'fixed', 'jittered' or 'random'.")
        valid = False
    if timemode not in ['periodic', 'drip-fixed', 'drip-jittered', 'drip-poisson']:
        print("CONFIG ERROR: timemode must be 'periodic', 'drip-fixed', 'drip-jittered' or 'drip-poisson'.")
        valid = False
    if interval <= 0:
        print("CONFIG ERROR: interval must be greater than 0.")
        valid = False
    if (supply['rangeMax']['rangeHigh'] < 0 or supply['rangeMax']['rangeLow'] < 0 or supply['rangeMin'][
        'rangeHigh'] < 0 or supply['rangeMin']['rangeLow'] < 0 or
            demand['rangeMax']['rangeHigh'] < 0 or demand['rangeMax']['rangeLow'] < 0 or demand['rangeMin'][
                'rangeHigh'] < 0 or demand['rangeMin']['rangeLow'] < 0):
        print("CONFIG ERROR: Supply range values must be greater than 0.")
        valid = False
    if (supply['rangeMax']['rangeHigh'] < supply['rangeMax']['rangeLow'] or demand['rangeMax']['rangeHigh'] <
            demand['rangeMax']['rangeLow'] or
            supply['rangeMin']['rangeHigh'] < supply['rangeMin']['rangeLow'] or demand['rangeMin']['rangeHigh'] <
            demand['rangeMin']['rangeLow']):
        print("CONFIG ERROR: rangeMax must be greater than or equal to rangeMin.")
        valid = False
    if numTrials < 1:
        print("CONFIG ERROR: numTrials must be greater than or equal to 1.")
        valid = False
    if numSchedulesPerRatio < 1:
        print("CONFIG ERROR: numSchedulesPerRatio must be greater than or equal to 1.")
        valid = False
    if numTrialsPerSchedule < 1:
        print("CONFIG ERROR: numTrialsPerSchedule must be greater than or equal to 1.")
        valid = False

    return valid
