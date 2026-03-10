"""Ордера через t_tech Sandbox API."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from t_tech.invest import Client, OrderDirection, OrderType, Quotation
from t_tech.invest.schemas import MoneyValue   # ← для SandboxPayIn
from config import config


def _units_nano(price: float):
    units = int(price)
    nano  = int(round((price - units) * 1_000_000_000))
    return units, nano


class SandboxOrders:

    def __init__(self):
        self.token = config.tinkoff.token
        self._account_id = None   # ← кэш

    def _get_or_create_account(self, client) -> str:
        if self._account_id:
            return self._account_id        # ← возвращаем кэш
        accounts = client.sandbox.get_sandbox_accounts().accounts
        if accounts:
            self._account_id = accounts[0].id
        else:
            resp = client.sandbox.open_sandbox_account()
            self._account_id = resp.account_id
        return self._account_id

    def top_up(self, amount: float = 1_000_000.0, currency: str = "rub") -> dict:
        """Пополнить баланс sandbox аккаунта."""
        with Client(self.token) as client:
            account_id = self._get_or_create_account(client)
            units, nano = _units_nano(amount)

            client.sandbox.sandbox_pay_in(
                account_id=account_id,
                amount=MoneyValue(
                    units=units,
                    nano=nano,
                    currency=currency,
                ),
            )

            # Проверяем итоговый баланс
            portfolio = client.sandbox.get_sandbox_portfolio(account_id=account_id)
            total = (
                float(portfolio.total_amount_portfolio.units)
                + portfolio.total_amount_portfolio.nano / 1e9
            )
            return {
                "account_id": account_id,
                "topped_up":  amount,
                "currency":   currency,
                "balance":    round(total, 2),
            }

    def post_order(
        self,
        figi:      str,
        direction: str,
        lots:      int,
        price:     float,
        order_id:  str,
    ) -> dict:
        dir_map = {
            "BUY":  OrderDirection.ORDER_DIRECTION_BUY,
            "SELL": OrderDirection.ORDER_DIRECTION_SELL,
        }

        with Client(self.token) as client:
            account_id = self._get_or_create_account(client)

            kwargs = dict(
                figi=figi,
                quantity=lots,
                direction=dir_map[direction],
                order_id=order_id,
                account_id=account_id,
            )

            if price == 0:
                kwargs["order_type"] = OrderType.ORDER_TYPE_MARKET
            else:
                u, n = _units_nano(price)
                kwargs["order_type"] = OrderType.ORDER_TYPE_LIMIT
                kwargs["price"]      = Quotation(units=u, nano=n)

            resp = client.sandbox.post_sandbox_order(**kwargs)

            exec_price = (
                float(resp.executed_order_price.units)
                + resp.executed_order_price.nano / 1e9
            )
            return {
                "order_id":   resp.order_id,
                "status":     str(resp.execution_report_status),
                "lots_exec":  resp.lots_executed,
                "price":      exec_price,
                "account_id": account_id,
            }

    def get_portfolio(self, account_id: str = None) -> dict:
        with Client(self.token) as client:
            if not account_id:
                account_id = self._get_or_create_account(client)

            resp  = client.sandbox.get_sandbox_portfolio(account_id=account_id)
            total = (
                float(resp.total_amount_portfolio.units)
                + resp.total_amount_portfolio.nano / 1e9
            )
            positions = [
                {
                    "figi":      p.figi,
                    "quantity":  float(p.quantity.units),
                    "avg_price": float(p.average_position_price.units)
                                 + p.average_position_price.nano / 1e9,
                }
                for p in resp.positions
            ]
            return {"total": round(total, 2), "positions": positions, "account_id": account_id}
