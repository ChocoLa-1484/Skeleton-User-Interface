import pandas as pd
import numpy as np

class PersonSelector:
    def __init__(self):
        self.selected_id = None

    def select(self, x=None, y=None, search_person_df=None):
        self.selected_id = None
        if search_person_df is None or search_person_df.empty:
            return None

        max_area = -1

        for _, row in search_person_df.iterrows():
            person_id = row['person_id']
            x1, y1, x2, y2 = map(int, row['bbox'])

            if x is not None and y is not None:
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    continue

            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                self.selected_id = person_id
        return self.selected_id

    def get_select_id(self):
        return self.selected_id

    def reset(self):
        self.selected_id = None
