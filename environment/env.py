import numpy as np
import enum as Enum
import pandas as pd
from collections import deque
from itertools import chain
from collections import deque
class State:
    def __init__(self, product_types_num,distr_warehouses_num,t,demand_history):
        #Total number of unique products
        self.product_types_num = product_types_num
        #Total number of stocks available at factory for each product
        self.factory_stocks = np.zeros(
            (self.product_types_num,),
            dtype=np.int32)
        #Total number of distribution warehouse
        self.distr_warehouses_num = distr_warehouses_num
        #stocks of each product available at each warehouse
        self.distr_warehouses_stocks = np.zeros(
            (self.distr_warehouses_num, self.product_types_num),
            dtype=np.int32)
        #self.T = T
        self.demand_history = demand_history
        self.t = t

    def to_array(self):
        return np.concatenate((
            self.factory_stocks,
            self.distr_warehouses_stocks.flatten(),
            np.hstack(list(chain(*chain(*self.demand_history)))),
            [self.t]))

    def stock_levels(self):
        return np.concatenate((
            self.factory_stocks,
            self.distr_warehouses_stocks.flatten()))

class Action:
    def __init__(self, product_types_num,distr_warehouses_num):
        # self.product_types_num = product_types_num
        # self.distr_warehouses_num = distr_warehouses_num
        #array representing the production levels for each product type.
        self.production_level = np.zeros((product_types_num,), dtype=np.int32)
        #shipping control for distributing stocks to different warehouses for each product type
        self.regular_shipped_stocks = np.zeros((distr_warehouses_num, product_types_num), dtype=np.int32)
        
class MakeEnv:
    def __init__(self, product_types_num, distr_warehouses, d_max,d_var, sale_prices, production_costs, storage_capacities, storage_costs, transportation_costs, penalty_costs_multiplier):
        self.product_types_num = product_types_num
        self.distr_warehouses = distr_warehouses
        self.T = 25
        self.d_max = np.array(d_max, np.int32)
        self.d_var=d_var
        #self.d_var = np.array(d_var, np.int32)
        self.sale_prices = np.array(sale_prices, np.int32)
        self.production_costs = np.array(production_costs, np.int32)
        self.storage_capacities = np.array(storage_capacities, np.int32)
        self.storage_costs = np.array(storage_costs, np.float32)
        self.transportation_costs = np.array(transportation_costs, np.float32)
        self.penalty_costs = .5*self.sale_prices
        self.reset()
    def reset(self,demand_history_len):
        self.demand_history=deque(maxlen=demand_history_len)
        for d in range(demand_history_len):
            self.demand_history.append(np.zeros(
                (self.distr_warehouses_num, self.product_types_num),
                dtype=np.int32))
        self.t = 0
        
        initial_state=State(self.product_types_num, self.distr_warehouses_num,
                     self.T, list(self.demand_history))
        return initial_state.to_array()

        
def demand(self, j, i, t):
        demand = np.round(
            self.d_max[i-1]/2 +
            self.d_max[i-1]/2*np.cos(4*np.pi*(2*j*i+t)/self.T) +
            np.random.randint(0, self.d_var[i-1]+1))
        return demand

# def initial_state(self):
#     return State(self.product_types_num, self.distr_warehouses_num,
#                      self.T, list(self.demand_history))

def step(self, state, action):
    demands = np.fromfunction(
        lambda j, i: self.demand(j+1, i+1, self.t),
        (self.distr_warehouses_num, self.product_types_num),
        dtype=np.int32)
    next_state = State(self.product_types_num, self.distr_warehouses_num,
                           self.T, list(self.demand_history))

    next_state.factory_stocks = np.minimum(
            np.subtract(np.add(state.factory_stocks,
                               action.production_level),
                        np.sum(action.shipped_stocks, axis=0)
                        ),
            self.storage_capacities[0]
        )

    for j in range(self.distr_warehouses_num):
            next_state.distr_warehouses_stocks[j] = np.minimum(
                np.subtract(np.add(state.distr_warehouses_stocks[j],
                                   action.shipped_stocks[j]),
                            demands[j]
                            ),
                self.storage_capacities[j+1]
            )
        # revenues
    total_revenues = np.dot(self.sale_prices,
                                np.sum(demands, axis=0))
        # production costs
    total_production_costs = np.dot(self.production_costs,
                                        action.production_level)
        # transportation costs
    total_transportation_costs = np.dot(
            self.transportation_costs.flatten(),
            action.shipped_stocks.flatten())
        # storage costs
    total_storage_costs = np.dot(
            self.storage_costs.flatten(),
            np.maximum(next_state.stock_levels(),
                       np.zeros(
                           ((self.distr_warehouses_num+1) *
                            self.product_types_num),
                           dtype=np.int32)
                       )
        )
        # penalty costs (minus sign because stock levels would be already
        # negative in case of unfulfilled demand)
    total_penalty_costs = -np.dot(
            self.penalty_costs,
            np.add(
                np.sum(
                    np.minimum(next_state.distr_warehouses_stocks,
                               np.zeros(
                                   (self.distr_warehouses_num,
                                    self.product_types_num),
                                   dtype=np.int32)
                               ),
                    axis=0),
                np.minimum(next_state.factory_stocks,
                           np.zeros(
                               (self.product_types_num,),
                               dtype=np.int32)
                           )
            )
        )
        # reward function
    reward = total_revenues - total_production_costs - \
            total_transportation_costs - total_storage_costs - \
            total_penalty_costs

        # the actual demand for the current time step will not be known until
        # the next time step. This implementation choice ensures that the agent
        # may benefit from learning the demand pattern so as to integrate a
        # sort of demand forecasting directly into the policy
    self.demand_history.append(demands)
        # actual time step value is not observed (for now)
    self.t += 1

    return next_state, reward, self.t == self.T-1
    # def initial_state(self):
    #     return State(self.product_types_num, self.distr_warehouses_num,
    #                  self.T, list(self.demand_history))
    
# def set_emergency_shipping(self, inventory_levels):
#     for product_type in range(self.product_types_num):
#         for warehouse in range(self.distr_warehouses_num):
#             if inventory_levels[warehouse, product_type] < self.inventory_threshold:
#                 self.emergency_shipped_stocks[warehouse, product_type] = self.inventory_threshold - inventory_levels[warehouse, product_type]
#             else:
#                 self.emergency_shipped_stocks[warehouse, product_type] = 0