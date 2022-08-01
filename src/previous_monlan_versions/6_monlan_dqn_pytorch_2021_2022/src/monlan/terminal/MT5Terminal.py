
import MetaTrader5 as mt5
import time
from copy import deepcopy

class MT5Terminal:
    def __init__(self, login, server, password):
        self.login = login
        self.server = server
        self.password = password
        self.terminal = mt5

        print("MetaTrader5 package author: ", self.terminal.__author__)
        print("MetaTrader5 package version: ", self.terminal.__version__)
        if not self.terminal.initialize(login=self.login, server=self.server, password=self.password):
            print("initialize() failed, error code =", mt5.last_error())
            quit()
        print(self.terminal.terminal_info())
        print(self.terminal.version())

        pass

    def __del__(self):
        self.terminal.shutdown()
        print( "mt5 server is off." )

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        timeframe = self.convert_timeframe( timeframe )
        return self.terminal.copy_rates_range(symbol, timeframe, date_from, date_to)

    def convert_timeframe(self, str_timeframe):
        timeframes_dict = {}
        timeframes_dict["M1"] = mt5.TIMEFRAME_M1
        timeframes_dict["M2"] = mt5.TIMEFRAME_M2
        timeframes_dict["M3"] = mt5.TIMEFRAME_M3
        timeframes_dict["M4"] = mt5.TIMEFRAME_M4
        timeframes_dict["M5"] = mt5.TIMEFRAME_M5
        timeframes_dict["M6"] = mt5.TIMEFRAME_M6
        timeframes_dict["M10"] = mt5.TIMEFRAME_M10
        timeframes_dict["M12"] = mt5.TIMEFRAME_M12
        timeframes_dict["M15"] = mt5.TIMEFRAME_M15
        timeframes_dict["M20"] = mt5.TIMEFRAME_M20
        timeframes_dict["M30"] = mt5.TIMEFRAME_M30
        timeframes_dict["H1"] = mt5.TIMEFRAME_H1
        timeframes_dict["H2"] = mt5.TIMEFRAME_H2
        timeframes_dict["H4"] = mt5.TIMEFRAME_H4
        timeframes_dict["H3"] = mt5.TIMEFRAME_H3
        timeframes_dict["H6"] = mt5.TIMEFRAME_H6
        timeframes_dict["H8"] = mt5.TIMEFRAME_H8
        timeframes_dict["H12"] = mt5.TIMEFRAME_H12
        timeframes_dict["D1"] = mt5.TIMEFRAME_D1
        timeframes_dict["W1"] = mt5.TIMEFRAME_W1
        timeframes_dict["MN1"] = mt5.TIMEFRAME_MN1

        return timeframes_dict[str_timeframe]

    def place_order( self, order_type, symbol, lot_size, stoploss_puncts, takeprofit_puncts ):
        symbol_info = self.terminal.symbol_info(symbol)
        if symbol_info is None:
            print(symbol, "not found, can not call order_check()")
            self.terminal.shutdown()
            quit()

        if not symbol_info.visible:
            print(symbol, "is not visible, trying to switch on")
            if not self.terminal.symbol_select(symbol, True):
                print("symbol_select({}}) failed, exit", symbol)
                self.terminal.shutdown()
                quit()

        point = self.terminal.symbol_info(symbol).point
        deviation = 100
        price = None
        stop_loss = None
        take_profit = None
        saved_order_type = deepcopy(order_type)
        if order_type == "buy":
            order_type = self.terminal.ORDER_TYPE_BUY
            price = self.terminal.symbol_info_tick(symbol).ask
            stop_loss = price - stoploss_puncts * point
            take_profit = price + takeprofit_puncts * point
        elif order_type == "sell":
            order_type = self.terminal.ORDER_TYPE_SELL
            price = self.terminal.symbol_info_tick(symbol).bid
            stop_loss = price + stoploss_puncts * point
            take_profit = price - takeprofit_puncts * point
        request = {
            "action": self.terminal.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": deviation,
            "magic": 234000,
            "comment": "open",
            "type_time": self.terminal.ORDER_TIME_GTC,
            "type_filling": self.terminal.ORDER_FILLING_RETURN,
        }

        # отправим торговый запрос
        result = self.terminal.order_send(request)
        result = self.check_open_result( order_type=saved_order_type, request=request, result=result,
                                       stoploss_puncts=stoploss_puncts, takeprofit_puncts=takeprofit_puncts)

        # проверим результат выполнения
        print("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, lot_size, price, deviation))
        if result.retcode != self.terminal.TRADE_RETCODE_DONE:
            print("order_send failed, retcode={}".format(result.retcode))
            # запросим результат в виде словаря и выведем поэлементно
            result_dict = result._asdict()
            for field in result_dict.keys():
                print("   {}={}".format(field, result_dict[field]))
                # если это структура торгового запроса, то выведем её тоже поэлементно
                if field == "request":
                    traderequest_dict = result_dict[field]._asdict()
                    for tradereq_filed in traderequest_dict:
                        print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
            print("shutdown() and quit")
            self.terminal.shutdown()
            quit()

        return result

    def check_limit_trigger(self, symbol):
        current_orders = self.terminal.positions_get(symbol=symbol)

        # recheck 2 times
        for i in range(2):
            if len(current_orders) == 0:
                time.sleep(1)
                current_orders = self.terminal.positions_get(symbol=symbol)
            if len(current_orders) != 0:
                break

        trigger = True
        if len(current_orders) > 0:
            trigger = False
        return trigger

    def close_order(self, order ):

        # создадим запрос на закрытие
        close_price = None
        close_type = None
        if order.request.type == self.terminal.ORDER_TYPE_BUY:
            close_price = self.terminal.symbol_info_tick(order.request.symbol).bid
            close_type = self.terminal.ORDER_TYPE_SELL
        elif order.request.type == self.terminal.ORDER_TYPE_SELL:
            close_price = self.terminal.symbol_info_tick(order.request.symbol).ask
            close_type = self.terminal.ORDER_TYPE_BUY

        request = {
            "action": order.request.action,
            "symbol": order.request.symbol,
            "volume": order.volume,
            "type": close_type,
            "position": order.order,
            "price": close_price,
            "deviation": order.request.deviation,
            "magic": 234000,
            "comment": "close",
            "type_time": order.request.type_time,
            "type_filling": order.request.type_filling,
        }
        # отправим торговый запрос
        result = self.terminal.order_send(request)
        result = self.check_close_result(request=request, order=order, result=result)


        # проверим результат выполнения
        print("close position #{}: sell {} {} lots at {} with deviation={} points".format(order.order, order.request.symbol, order.request.volume,
            close_price, order.request.deviation))
        if result.retcode != self.terminal.TRADE_RETCODE_DONE:
            print("order_send failed, retcode={}".format(result.retcode))
            print("   result", result)
        else:
            print("position #{} closed, {}".format(order.order, result))
            # запросим результат в виде словаря и выведем поэлементно
            result_dict = result._asdict()
            for field in result_dict.keys():
                print("   {}={}".format(field, result_dict[field]))
                # если это структура торгового запроса, то выведем её тоже поэлементно
                if field == "request":
                    traderequest_dict = result_dict[field]._asdict()
                    for tradereq_filed in traderequest_dict:
                        print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))

        return True

    def check_open_result(self, order_type, request, result, stoploss_puncts, takeprofit_puncts):
        # process fails
        if result.retcode in [10004, 10021]:
            order_resend = False
            for i in range(6):
                sleep_sec = 10
                print("Catched fail code: {}. Waiting {} seconds and resend order.".format(result.retcode, sleep_sec))
                time.sleep(sleep_sec)
                saved_order_type = deepcopy(order_type)
                request = self.remake_open_request(saved_order_type, request, stoploss_puncts, takeprofit_puncts)
                result = self.terminal.order_send(request)
                if result.retcode not in [10004, 10021]:
                    order_resend = True
                    print("Order resent.")
                    break
            if order_resend == False:
                print("Fail to resend open order. Stop execution.")
                self.terminal.shutdown()
                quit()
        return result

    def check_close_result(self, request, order, result):
        # process fails
        if result.retcode in [10004, 10021]:
            order_resend = False
            for i in range(6):
                sleep_sec = 10
                print("Catched fail code: {}. Waiting {} seconds and resend order.".format(result.retcode, sleep_sec))
                time.sleep(sleep_sec)
                request = self.remake_close_request(order)
                result = self.terminal.order_send(request)
                if result.retcode not in [10004, 10021]:
                    order_resend = True
                    print("Order resent.")
                    break
            if order_resend == False:
                print("Fail to resend order. Stop execution.")
                self.terminal.shutdown()
                quit()
        return result

    def remake_open_request(self, order_type, request, stoploss_puncts, takeprofit_puncts):

        symbol = request["symbol"]
        lot_size = request["volume"]

        point = self.terminal.symbol_info(symbol).point
        deviation = 100
        price = None
        stop_loss = None
        take_profit = None
        saved_order_type = deepcopy(order_type)
        if saved_order_type == "buy":
            saved_order_type = self.terminal.ORDER_TYPE_BUY
            price = self.terminal.symbol_info_tick(symbol).ask
            stop_loss = price - stoploss_puncts * point
            take_profit = price + takeprofit_puncts * point
        elif saved_order_type == "sell":
            saved_order_type = self.terminal.ORDER_TYPE_SELL
            price = self.terminal.symbol_info_tick(symbol).bid
            stop_loss = price + stoploss_puncts * point
            take_profit = price - takeprofit_puncts * point
        request = {
            "action": self.terminal.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": saved_order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": deviation,
            "magic": 234000,
            "comment": "open",
            "type_time": self.terminal.ORDER_TIME_GTC,
            "type_filling": self.terminal.ORDER_FILLING_RETURN,
        }

        return request

    def remake_close_request(self, order):

        close_price = None
        close_type = None
        order = deepcopy(order)
        if order.request.type == self.terminal.ORDER_TYPE_BUY:
            close_price = self.terminal.symbol_info_tick(order.request.symbol).bid
            close_type = self.terminal.ORDER_TYPE_SELL
        elif order.request.type == self.terminal.ORDER_TYPE_SELL:
            close_price = self.terminal.symbol_info_tick(order.request.symbol).ask
            close_type = self.terminal.ORDER_TYPE_BUY

        request = {
            "action": order.request.action,
            "symbol": order.request.symbol,
            "volume": order.volume,
            "type": close_type,
            "position": order.order,
            "price": close_price,
            "deviation": order.request.deviation,
            "magic": 234000,
            "comment": "close",
            "type_time": order.request.type_time,
            "type_filling": order.request.type_filling,
        }

        return request