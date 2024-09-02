

import importlib
import pkgutil

print("Loading modules in base package...")

for module_info in pkgutil.iter_modules(__path__):
    module_name = module_info.name
    print(f"Importing module: {module_name}")
    importlib.import_module(f'.{module_name}', package=__name__)

print("Modules loaded.")

from .memory import show
from .formulate_coding_AST import load_df
from .formulate_coding_AST import f_coding
from .my_backtesting import Backtesting
from .my_backtesting import Decile
from .my_df_func import*
from .my_os_func import*
from .get_tushare_data import*
from .binance_trade_client import*
from .mapping_columns_name import trans_old2_yd_data