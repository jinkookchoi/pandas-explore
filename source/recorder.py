import uuid
import pandas as pd
from datetime import date, datetime, timedelta
from pytz import timezone
from tinydb import TinyDB, Query

class Recorder:
    """모듈 학습/테스트 이력을 기록하는 모듈입니다.

    JSON 파일 데이터베이스를 이용하여 훈련 내용을 기록하거나
    특정 기록 내용을 조회할 수 있습니다.
    """
    def __init__(self, history_fn, metric='auc'):
        self.db = TinyDB(history_fn)
        self.metric = metric
        self.template = {'features': None,
                        'n_features_in': None,
                        self.metric: None,
                        'tag': None,
                        'user': None,
                        'elapsed_time': None,
                        'model': None,
                        'hyper_parameters': None}

    def __len__(self):
        return len(pd.DataFrame(self.db.all()))

    def insert(self, record, bypass=False):
        record['uid'] = str(uuid.uuid4())
        record['datetime'] = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
        self.db.insert(record)
        
        if bypass:
            return record

    def get_best(self, n=1):
        if len(self) > 0:
            return pd.DataFrame(self.db.all()).sort_values(by=self.metric, ascending=False).head(n)
        else:
            print('Empty Records')
            return None

    def get_recent(self, n=1, datetime='datetime', reverse=False):
        if len(self) > 0:
            return pd.DataFrame(self.db.all()).sort_values(by=datetime, ascending=reverse).head(n)
        else:
            print('Empty Records')
            return None

    def get_all(self, dataframe=True):
        if dataframe:
            return pd.DataFrame(self.db.all())
        else:
            return self.db.all()

    def query(self, key, value):
        if len(self) > 0:
            return pd.DataFrame(self.db.all()).query(f'{key}.str.contains("{value}")')
        else:
            print('Empty Records')
            return None