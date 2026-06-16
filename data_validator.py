import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ValidationError:
    error_type: str
    table: str
    column: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        col = f".{self.column}" if self.column else ""
        return f"[{self.error_type}] {self.table}{col}: {self.message}"


@dataclass
class ValidationResult:
    table: str
    passed: bool
    errors: List[ValidationError] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"=== {self.table} [{status}] ==="]
        for err in self.errors:
            lines.append(f"  {err}")
        return "\n".join(lines)


@dataclass
class ForeignKeyRule:
    fk_columns: List[str]
    ref_table: str
    ref_columns: List[str]


class DataValidator:
    def __init__(self, tables: Dict[str, pd.DataFrame]):
        self.tables = tables

    def validate_primary_key(self, table_name: str, pk_columns: List[str]) -> List[ValidationError]:
        errors: List[ValidationError] = []
        df = self.tables.get(table_name)
        if df is None:
            errors.append(ValidationError(
                error_type="TABLE_NOT_FOUND",
                table=table_name,
                message=f"表 '{table_name}' 不存在"
            ))
            return errors

        for col in pk_columns:
            if col not in df.columns:
                errors.append(ValidationError(
                    error_type="COLUMN_NOT_FOUND",
                    table=table_name,
                    column=col,
                    message=f"主键列 '{col}' 不存在"
                ))
        if errors:
            return errors

        duplicated = df[df.duplicated(subset=pk_columns, keep=False)]
        if not duplicated.empty:
            dup_count = len(duplicated)
            dup_values = duplicated[pk_columns].drop_duplicates().head(10).to_dict('records')
            errors.append(ValidationError(
                error_type="PK_DUPLICATE",
                table=table_name,
                column=",".join(pk_columns),
                message=f"发现 {dup_count} 条主键重复记录",
                details={"duplicate_count": dup_count, "sample_duplicates": dup_values}
            ))

        return errors

    def validate_required_columns(self, table_name: str, required_columns: List[str]) -> List[ValidationError]:
        errors: List[ValidationError] = []
        df = self.tables.get(table_name)
        if df is None:
            errors.append(ValidationError(
                error_type="TABLE_NOT_FOUND",
                table=table_name,
                message=f"表 '{table_name}' 不存在"
            ))
            return errors

        for col in required_columns:
            if col not in df.columns:
                errors.append(ValidationError(
                    error_type="COLUMN_NOT_FOUND",
                    table=table_name,
                    column=col,
                    message=f"必填列 '{col}' 不存在"
                ))
                continue

            null_count = df[col].isna().sum()
            if null_count > 0:
                null_indices = df[df[col].isna()].index.tolist()[:10]
                errors.append(ValidationError(
                    error_type="REQUIRED_NULL",
                    table=table_name,
                    column=col,
                    message=f"存在 {null_count} 条空值记录",
                    details={"null_count": int(null_count), "sample_indices": null_indices}
                ))

        return errors

    def validate_foreign_key(self, table_name: str, fk_rule: ForeignKeyRule) -> List[ValidationError]:
        errors: List[ValidationError] = []
        df = self.tables.get(table_name)
        ref_df = self.tables.get(fk_rule.ref_table)

        if df is None:
            errors.append(ValidationError(
                error_type="TABLE_NOT_FOUND",
                table=table_name,
                message=f"表 '{table_name}' 不存在"
            ))
            return errors

        if ref_df is None:
            errors.append(ValidationError(
                error_type="TABLE_NOT_FOUND",
                table=fk_rule.ref_table,
                message=f"引用表 '{fk_rule.ref_table}' 不存在"
            ))
            return errors

        for col in fk_rule.fk_columns:
            if col not in df.columns:
                errors.append(ValidationError(
                    error_type="COLUMN_NOT_FOUND",
                    table=table_name,
                    column=col,
                    message=f"外键列 '{col}' 不存在"
                ))
        for col in fk_rule.ref_columns:
            if col not in ref_df.columns:
                errors.append(ValidationError(
                    error_type="COLUMN_NOT_FOUND",
                    table=fk_rule.ref_table,
                    column=col,
                    message=f"引用列 '{col}' 不存在"
                ))
        if errors:
            return errors

        if len(fk_rule.fk_columns) != len(fk_rule.ref_columns):
            errors.append(ValidationError(
                error_type="FK_COLUMN_MISMATCH",
                table=table_name,
                message=f"外键列数量 ({len(fk_rule.fk_columns)}) 与引用列数量 ({len(fk_rule.ref_columns)}) 不匹配"
            ))
            return errors

        fk_df = df[fk_rule.fk_columns].copy()
        ref_df_cols = ref_df[fk_rule.ref_columns].copy()
        ref_df_cols.columns = fk_rule.fk_columns

        has_null = fk_df.isna().any(axis=1)
        fk_non_null = fk_df[~has_null]

        merged = fk_non_null.merge(
            ref_df_cols.assign(_exists=True),
            on=fk_rule.fk_columns,
            how="left"
        )
        missing = merged[merged["_exists"].isna()]

        if not missing.empty:
            missing_count = len(missing)
            missing_values = missing[fk_rule.fk_columns].drop_duplicates().head(10).to_dict('records')
            errors.append(ValidationError(
                error_type="FK_NOT_FOUND",
                table=table_name,
                column=",".join(fk_rule.fk_columns),
                message=f"存在 {missing_count} 条外键在 '{fk_rule.ref_table}' 中未找到",
                details={
                    "missing_count": missing_count,
                    "referenced_table": fk_rule.ref_table,
                    "sample_missing": missing_values
                }
            ))

        return errors

    def validate_table(
        self,
        table_name: str,
        pk_columns: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        fk_rules: Optional[List[ForeignKeyRule]] = None
    ) -> ValidationResult:
        errors: List[ValidationError] = []

        if pk_columns:
            errors.extend(self.validate_primary_key(table_name, pk_columns))

        if required_columns:
            errors.extend(self.validate_required_columns(table_name, required_columns))

        if fk_rules:
            for rule in fk_rules:
                errors.extend(self.validate_foreign_key(table_name, rule))

        return ValidationResult(
            table=table_name,
            passed=len(errors) == 0,
            errors=errors
        )

    def validate_all(self, rules: Dict[str, Dict[str, Any]]) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        for table_name, rule in rules.items():
            fk_rules = rule.get("fk_rules", [])
            fk_rule_objects = [
                ForeignKeyRule(**fk) if isinstance(fk, dict) else fk
                for fk in fk_rules
            ]
            result = self.validate_table(
                table_name=table_name,
                pk_columns=rule.get("pk"),
                required_columns=rule.get("required"),
                fk_rules=fk_rule_objects
            )
            results.append(result)
        return results
