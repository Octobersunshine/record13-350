import pandas as pd
from data_validator import DataValidator, ForeignKeyRule


def main():
    users = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "username": ["alice", "bob", "charlie", "david", "eve"],
        "email": ["alice@example.com", "bob@example.com", None, "david@example.com", "eve@example.com"],
        "dept_id": [101, 102, 101, 999, None]
    })

    departments = pd.DataFrame({
        "dept_id": [101, 102, 103],
        "dept_name": ["Engineering", "Marketing", "Sales"],
        "manager_id": [1, 2, 2]
    })

    orders = pd.DataFrame({
        "order_id": [1001, 1002, 1003, 1003, 1004],
        "user_id": [1, 2, 3, 3, 6],
        "amount": [99.9, 49.5, None, 150.0, 200.0],
        "status": ["paid", "pending", "paid", "paid", "shipped"]
    })

    tables = {
        "users": users,
        "departments": departments,
        "orders": orders
    }

    validator = DataValidator(tables)

    rules = {
        "departments": {
            "pk": ["dept_id"],
            "required": ["dept_id", "dept_name"]
        },
        "users": {
            "pk": ["user_id"],
            "required": ["user_id", "username", "email"],
            "fk_rules": [
                ForeignKeyRule(
                    fk_columns=["dept_id"],
                    ref_table="departments",
                    ref_columns=["dept_id"]
                )
            ]
        },
        "orders": {
            "pk": ["order_id"],
            "required": ["order_id", "user_id", "amount"],
            "fk_rules": [
                ForeignKeyRule(
                    fk_columns=["user_id"],
                    ref_table="users",
                    ref_columns=["user_id"]
                )
            ]
        }
    }

    results = validator.validate_all(rules)

    print("=" * 60)
    print("数据完整性校验报告")
    print("=" * 60)

    total_pass = 0
    total_fail = 0

    for result in results:
        print()
        print(result)
        if result.passed:
            total_pass += 1
        else:
            total_fail += 1

    print()
    print("=" * 60)
    print(f"汇总: {total_pass} 个表通过, {total_fail} 个表失败")
    print("=" * 60)

    print("\n--- 详细错误示例 (orders 表) ---")
    orders_result = [r for r in results if r.table == "orders"][0]
    for err in orders_result.errors:
        print(f"\n错误类型: {err.error_type}")
        print(f"消息: {err.message}")
        if err.details:
            print(f"详情: {err.details}")


if __name__ == "__main__":
    main()
