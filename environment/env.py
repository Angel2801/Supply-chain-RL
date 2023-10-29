import numpy as np
import enum as Enum
import pandas as pd
from collections import deque
from itertools import chain

class State:
    def __init__(self, product_types_num, distr_warehouses_num, T,
                 demand_history, t=0):
        self.product_types_num = product_types_num
        self.factory_stocks = np.zeros(
            (self.product_types_num,),
            dtype=np.int32)
        self.distr_warehouses_num = distr_warehouses_num
        self.distr_warehouses_stocks = np.zeros(
            (self.distr_warehouses_num, self.product_types_num),
            dtype=np.int32)
        self.T = T
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
    def __init__(self, product_types_num,inventory_threshold=10):
        # self.product_types_num = product_types_num
        # self.distr_warehouses_num = distr_warehouses_num
        # self.inventory_threshold = inventory_threshold

        self.production_level = np.zeros((product_types_num,), dtype=np.int32)
        self.regular_shipped_stocks = np.zeros((distr_warehouses_num, product_types_num), dtype=np.int32)
        #self.emergency_shipped_stocks = np.zeros((distr_warehouses_num, product_types_num), dtype=np.int32)
        
class MakeEnv:
    def __init__(self, product_types, distr_warehouses, T=25, d_max, d_var, sale_prices, production_costs, storage_capacities, storage_costs, transportation_costs, penalty_costs_multiplier):
        self.product_types = product_types
        self.distr_warehouses = distr_warehouses
        self.T = T

        self.d_max = np.array(d_max, np.int32)
        self.d_var = np.array(d_var, np.int32)
        self.sale_prices = np.array(sale_prices, np.int32)
        self.production_costs = np.array(production_costs, np.int32)
        self.storage_capacities = np.array(storage_capacities, np.int32)
        self.storage_costs = np.array(storage_costs, np.float32)
        self.transportation_costs = np.array(transportation_costs, np.float32)
        
    def calculate_dynamic_multiplier(self, product_type):
        product_data = self.data[self.data["ProductType"] == product_type]
        demand_variation_std = product_data["DemandVariation"].std()
        multiplier = demand_variation_std * 0.1
        return multiplier
    def calculate_dynamic_penalty_costs(self, product_type):
        multiplier = self.calculate_dynamic_multiplier(product_type)
        product_data = self.data[self.data["ProductType"] == product_type]
        penalty_costs = product_data["LostSalesCost"] * multiplier
        return penalty_costs
    self.penalty_costs = calculate_dynamic_penalty_costs(product_type)
        
    def demand(self, j, i, t):
        max_demand = self.d_max[i - 1]
        demand_variation = self.d_var[i - 1]
        half_max_demand = max_demand / 2
        pi_term = 4 * np.pi / self.T
        demand = np.round(
        half_max_demand + half_max_demand * np.cos(pi_term * (2 * j * i + t)) +
        np.random.randint(0, demand_variation + 1, len(j)))
        return demand
    
    def initial_state(self):
        return State(self.product_types_num, self.distr_warehouses_num,
                     self.T, list(self.demand_history))

    
# def set_emergency_shipping(self, inventory_levels):
#     for product_type in range(self.product_types_num):
#         for warehouse in range(self.distr_warehouses_num):
#             if inventory_levels[warehouse, product_type] < self.inventory_threshold:
#                 self.emergency_shipped_stocks[warehouse, product_type] = self.inventory_threshold - inventory_levels[warehouse, product_type]
#             else:
#                 self.emergency_shipped_stocks[warehouse, product_type] = 0