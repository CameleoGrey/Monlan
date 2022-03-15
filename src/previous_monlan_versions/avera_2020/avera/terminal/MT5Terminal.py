
import MetaTrader5 as mt5
import time
from copy import deepcopy

class MT5Terminal:
    def __init__(self, login=50581877, server="Alpari-MT5-Demo", password="6v2v3ve9N"):
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
        timeframe = self.convertTimeframe( timeframe )
        return self.terminal.copy_rates_range(symbol, timeframe, date_from, date_to)

    def convertTimeframe(self, strTimeframe):
        timeframesDict = {}
        timeframesDict["M1"] = mt5.TIMEFRAME_M1
        timeframesDict["M2"] = mt5.TIMEFRAME_M2
        timeframesDict["M3"] = mt5.TIMEFRAME_M3
        timeframesDict["M4"] = mt5.TIMEFRAME_M4
        timeframesDict["M5"] = mt5.TIMEFRAME_M5
        timeframesDict["M6"] = mt5.TIMEFRAME_M6
        timeframesDict["M10"] = mt5.TIMEFRAME_M10
        timeframesDict["M12"] = mt5.TIMEFRAME_M12
        timeframesDict["M15"] = mt5.TIMEFRAME_M15
        timeframesDict["M20"] = mt5.TIMEFRAME_M20
        timeframesDict["M30"] = mt5.TIMEFRAME_M30
        timeframesDict["H1"] = mt5.TIMEFRAME_H1
        timeframesDict["H2"] = mt5.TIMEFRAME_H2
        timeframesDict["H4"] = mt5.TIMEFRAME_H4
        timeframesDict["H3"] = mt5.TIMEFRAME_H3
        timeframesDict["H6"] = mt5.TIMEFRAME_H6
        timeframesDict["H8"] = mt5.TIMEFRAME_H8
        timeframesDict["H12"] = mt5.TIMEFRAME_H12
        timeframesDict["D1"] = mt5.TIMEFRAME_D1
        timeframesDict["W1"] = mt5.TIMEFRAME_W1
        timeframesDict["MN1"] = mt5.TIMEFRAME_MN1

        return timeframesDict[strTimeframe]

    def placeOrder( self, orderType, symbol, lotSize, stoplossPuncts, takeprofitPuncts ):
        symbol_info = self.terminal.symbol_info(symbol)
        if symbol_info is None:
            print(symbol, "not found, can not call order_check()")
            self.terminal.shutdown()
            quit()

        # если символ недоступен в MarketWatch, добавим его
        if not symbol_info.visible:
            print(symbol, "is not visible, trying to switch on")
            if not self.terminal.symbol_select(symbol, True):
                print("symbol_select({}}) failed, exit", symbol)
                self.terminal.shutdown()
                quit()

        point = self.terminal.symbol_info(symbol).point
        deviation = 100
        price = None
        stopLoss = None
        takeProfit = None
        savedOrderType = deepcopy(orderType)
        if orderType == "buy":
            orderType = self.terminal.ORDER_TYPE_BUY
            price = self.terminal.symbol_info_tick(symbol).ask
            stopLoss = price - stoplossPuncts * point
            takeProfit = price + takeprofitPuncts * point
        elif orderType == "sell":
            orderType = self.terminal.ORDER_TYPE_SELL
            price = self.terminal.symbol_info_tick(symbol).bid
            stopLoss = price + stoplossPuncts * point
            takeProfit = price - takeprofitPuncts * point
        request = {
            "action": self.terminal.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lotSize,
            "type": orderType,
            "price": price,
            "sl": stopLoss,
            "tp": takeProfit,
            "deviation": deviation,
            "magic": 234000,
            "comment": "open",
            "type_time": self.terminal.ORDER_TIME_GTC,
            "type_filling": self.terminal.ORDER_FILLING_RETURN,
        }

        # отправим торговый запрос
        result = self.terminal.order_send(request)
        result = self.checkOpenResult( orderType=savedOrderType, request=request, result=result,
                                       stoplossPuncts=stoplossPuncts, takeprofitPuncts=takeprofitPuncts)

        # проверим результат выполнения
        print("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, lotSize, price, deviation))
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

    def checkLimitTrigger(self, symbol):
        currentOrders = self.terminal.positions_get(symbol=symbol)

        # recheck 2 times
        for i in range(2):
            if len(currentOrders) == 0:
                time.sleep(1)
                currentOrders = self.terminal.positions_get(symbol=symbol)
            if len(currentOrders) != 0:
                break

        trigger = True
        if len(currentOrders) > 0:
            trigger = False
        return trigger

    def closeOrder(self, order ):

        # создадим запрос на закрытие
        closePrice = None
        closeType = None
        if order.request.type == self.terminal.ORDER_TYPE_BUY:
            closePrice = self.terminal.symbol_info_tick(order.request.symbol).bid
            closeType = self.terminal.ORDER_TYPE_SELL
        elif order.request.type == self.terminal.ORDER_TYPE_SELL:
            closePrice = self.terminal.symbol_info_tick(order.request.symbol).ask
            closeType = self.terminal.ORDER_TYPE_BUY

        request = {
            "action": order.request.action,
            "symbol": order.request.symbol,
            "volume": order.volume,
            "type": closeType,
            "position": order.order,
            "price": closePrice,
            "deviation": order.request.deviation,
            "magic": 234000,
            "comment": "close",
            "type_time": order.request.type_time,
            "type_filling": order.request.type_filling,
        }
        # отправим торговый запрос
        result = self.terminal.order_send(request)
        result = self.checkCloseResult(request=request, order=order, result=result)


        # проверим результат выполнения
        print("close position #{}: sell {} {} lots at {} with deviation={} points".format(order.order, order.request.symbol, order.request.volume,
            closePrice, order.request.deviation))
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

    def checkOpenResult(self, orderType, request, result, stoplossPuncts, takeprofitPuncts):
        # process fails
        if result.retcode in [10004, 10021]:
            orderResend = False
            for i in range(6):
                sleepSec = 10
                print("Catched fail code: {}. Waiting {} seconds and resend order.".format(result.retcode, sleepSec))
                time.sleep(sleepSec)
                savedOrderType = deepcopy(orderType)
                request = self.remakeOpenRequest(savedOrderType, request, stoplossPuncts, takeprofitPuncts)
                result = self.terminal.order_send(request)
                if result.retcode not in [10004, 10021]:
                    orderResend = True
                    print("Order resent.")
                    break
            if orderResend == False:
                print("Fail to resend open order. Stop execution.")
                self.terminal.shutdown()
                quit()
        return result

    def checkCloseResult(self, request, order, result):
        # process fails
        if result.retcode in [10004, 10021]:
            orderResend = False
            for i in range(6):
                sleepSec = 10
                print("Catched fail code: {}. Waiting {} seconds and resend order.".format(result.retcode, sleepSec))
                time.sleep(sleepSec)
                request = self.remakeCloseRequest(order)
                result = self.terminal.order_send(request)
                if result.retcode not in [10004, 10021]:
                    orderResend = True
                    print("Order resent.")
                    break
            if orderResend == False:
                print("Fail to resend order. Stop execution.")
                self.terminal.shutdown()
                quit()
        return result

    def remakeOpenRequest(self, orderType, request, stoplossPuncts, takeprofitPuncts):

        symbol = request["symbol"]
        lotSize = request["volume"]

        point = self.terminal.symbol_info(symbol).point
        deviation = 100
        price = None
        stopLoss = None
        takeProfit = None
        savedOrderType = deepcopy(orderType)
        if savedOrderType == "buy":
            savedOrderType = self.terminal.ORDER_TYPE_BUY
            price = self.terminal.symbol_info_tick(symbol).ask
            stopLoss = price - stoplossPuncts * point
            takeProfit = price + takeprofitPuncts * point
        elif savedOrderType == "sell":
            savedOrderType = self.terminal.ORDER_TYPE_SELL
            price = self.terminal.symbol_info_tick(symbol).bid
            stopLoss = price + stoplossPuncts * point
            takeProfit = price - takeprofitPuncts * point
        request = {
            "action": self.terminal.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lotSize,
            "type": savedOrderType,
            "price": price,
            "sl": stopLoss,
            "tp": takeProfit,
            "deviation": deviation,
            "magic": 234000,
            "comment": "open",
            "type_time": self.terminal.ORDER_TIME_GTC,
            "type_filling": self.terminal.ORDER_FILLING_RETURN,
        }

        return request

    def remakeCloseRequest(self, order):

        # создадим запрос на закрытие
        closePrice = None
        closeType = None
        order = deepcopy(order)
        if order.request.type == self.terminal.ORDER_TYPE_BUY:
            closePrice = self.terminal.symbol_info_tick(order.request.symbol).bid
            closeType = self.terminal.ORDER_TYPE_SELL
        elif order.request.type == self.terminal.ORDER_TYPE_SELL:
            closePrice = self.terminal.symbol_info_tick(order.request.symbol).ask
            closeType = self.terminal.ORDER_TYPE_BUY

        request = {
            "action": order.request.action,
            "symbol": order.request.symbol,
            "volume": order.volume,
            "type": closeType,
            "position": order.order,
            "price": closePrice,
            "deviation": order.request.deviation,
            "magic": 234000,
            "comment": "close",
            "type_time": order.request.type_time,
            "type_filling": order.request.type_filling,
        }

        return request