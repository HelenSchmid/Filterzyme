from __future__ import annotations
import pandas as pd
from typing import Union
from pathlib import Path

class Pipeline():
    def __init__(self, *steps: Step):
        self.steps = list(steps)

    def execute(self, input_data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        for step in self.steps:
            input_data = step.execute(input_data)
        return input_data

    def __rlshift__(self, other: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        return self.execute(other)

    def __rshift__(self, other: Step) -> 'Pipeline':
        return Pipeline(*self.steps, other)

'''
class Pipeline():
    
    def __init__(self, *steps: Step):
        self.steps = list(steps)
        
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Execute some shit.
        """      
        for step in self.steps:
            df = step.execute(df)
        return df
    
    def __rlshift__(self, other: pd.DataFrame) -> pd.DataFrame:
        return self.execute(other)
        
class Pipeline:
    def __init__(self, *steps: Step):
        self.steps = list(steps)

    def execute(self, input_data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        for step in self.steps:
            input_data = step.execute(input_data)
        return input_data

    def __rlshift__(self, other: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        return self.execute(other)
'''

class Step():
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Execute some shit """ 
        return df
    
    def __rshift__(self, other: Step) -> Step:
        return Pipeline(self, other)
        
    def __rlshift__(self, other: pd.DataFrame) -> pd.DataFrame:
        """
        Overriding the right shift operator to allow for the pipeline to be executed.
        """
        return self.execute(other)
    
class Step:
    def execute(self, input_data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        return pd.DataFrame()

    def __rshift__(self, other: 'Step') -> 'Pipeline':
        return Pipeline(self, other)

    def __rlshift__(self, other: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        return self.execute(other)
