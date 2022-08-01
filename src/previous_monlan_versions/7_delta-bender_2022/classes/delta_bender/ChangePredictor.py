
from classes.delta_bender.FeatGenMeta import FeatGenMeta
from classes.multipurpose.utils import *

class ChangePredictor():
    def __init__(self, use_heiken_ashi,
                 ema_period, n_price_feats, m_cdv_feats,
                 model, encoder=None, batch_size=128):

        self.feat_gen = FeatGenMeta( use_heiken_ashi )

        self.ema_period = ema_period
        self.n_price_feats = n_price_feats
        self.m_cdv_feats = m_cdv_feats

        self.encoder = encoder
        self.batch_size = batch_size

        self.model = model

        pass

    def extract_feats_(self, df):
        df = df.copy()
        x = self.feat_gen.transform(df,
                                    ema_period=self.ema_period,
                                    n_price_feats=self.n_price_feats,
                                    m_cdv_feats=self.m_cdv_feats)

        if self.encoder is not None:
            x = x.reshape((1,) + x.shape + (1,))
            x = self.encoder.predict(x, batch_size=self.batch_size)
        else:
            x = x.reshape((1,) + x.shape)

        return x

    def predict(self, df):

        x = self.extract_feats_( df )
        y = self.model.predict( x )

        return y

    def predict_proba(self, df):

        x = self.extract_feats_(df)
        y = self.model.predict_proba(x)
        y = y[0]

        return y