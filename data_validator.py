import json
import pandas as pd
from datetime import datetime
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "table": self.table,
            "column": self.column,
            "message": self.message,
            "details": self.details,
        }


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "passed": self.passed,
            "errors": [e.to_dict() for e in self.errors],
        }


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

        pk_values = df[pk_columns].itertuples(index=False, name=None)
        seen: set = set()
        dup_values: set = set()
        dup_count = 0
        for val in pk_values:
            if val in seen:
                dup_count += 1
                dup_values.add(val)
            else:
                seen.add(val)

        if dup_count > 0:
            sample = list(dup_values)[:10]
            errors.append(ValidationError(
                error_type="PK_DUPLICATE",
                table=table_name,
                column=",".join(pk_columns),
                message=f"发现 {dup_count} 条主键重复记录",
                details={
                    "duplicate_count": dup_count,
                    "sample_duplicates": [
                        dict(zip(pk_columns, v)) for v in sample
                    ]
                }
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


class ReportExporter:
    def __init__(self, results: List[ValidationResult]):
        self.results = results

    def _build_report_data(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        total_errors = sum(len(r.errors) for r in self.results)
        return {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_tables": total,
                "passed": passed,
                "failed": failed,
                "total_errors": total_errors,
            },
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._build_report_data(), indent=indent, ensure_ascii=False)

    def export_json(self, filepath: str, indent: int = 2) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent))

    def to_html(self) -> str:
        data = self._build_report_data()
        summary = data["summary"]
        rows_html = []
        for r in data["results"]:
            status_cls = "pass" if r["passed"] else "fail"
            status_text = "PASS" if r["passed"] else "FAIL"
            errors_html = ""
            if r["errors"]:
                error_items = []
                for e in r["errors"]:
                    details_str = ""
                    if e["details"]:
                        detail_rows = "".join(
                            f'<tr><td class="detail-key">{k}</td><td class="detail-val">{v}</td></tr>'
                            for k, v in e["details"].items()
                        )
                        details_str = (
                            '<table class="detail-table">'
                            + detail_rows
                            + "</table>"
                        )
                    error_items.append(
                        f'<tr><td class="err-type">{e["error_type"]}</td>'
                        f'<td>{e["column"] or ""}</td>'
                        f'<td>{e["message"]}</td>'
                        f'<td>{details_str}</td></tr>'
                    )
                errors_html = (
                    '<table class="error-table">'
                    '<tr><th>错误类型</th><th>列</th><th>消息</th><th>详情</th></tr>'
                    + "".join(error_items)
                    + "</table>"
                )
            else:
                errors_html = '<span class="no-error">无错误</span>'
            rows_html.append(
                f'<tr><td>{r["table"]}</td>'
                f'<td class="{status_cls}">{status_text}</td>'
                f'<td>{errors_html}</td></tr>'
            )

        return (
            "<!DOCTYPE html>"
            '<html lang="zh-CN"><head><meta charset="UTF-8">'
            "<title>数据完整性校验报告</title>"
            "<style>"
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
            "margin:40px;background:#f5f7fa;color:#333}"
            "h1{color:#1a1a2e;border-bottom:2px solid #4361ee;padding-bottom:8px}"
            ".summary{display:flex;gap:20px;margin:20px 0}"
            ".summary-card{background:#fff;border-radius:8px;padding:16px 24px;"
            "box-shadow:0 2px 8px rgba(0,0,0,.08);text-align:center;min-width:120px}"
            ".summary-card .num{font-size:28px;font-weight:700}"
            ".summary-card .label{font-size:13px;color:#666;margin-top:4px}"
            ".summary-card.total .num{color:#4361ee}"
            ".summary-card.pass .num{color:#2ecc71}"
            ".summary-card.fail .num{color:#e74c3c}"
            ".summary-card.err .num{color:#f39c12}"
            "table.main-table{width:100%;border-collapse:collapse;background:#fff;"
            "border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08)}"
            "table.main-table th{background:#1a1a2e;color:#fff;padding:12px 16px;text-align:left}"
            "table.main-table td{padding:12px 16px;border-bottom:1px solid #eee;vertical-align:top}"
            "table.main-table tr:last-child td{border-bottom:none}"
            ".pass{color:#2ecc71;font-weight:700}"
            ".fail{color:#e74c3c;font-weight:700}"
            "table.error-table{width:100%;border-collapse:collapse;margin-top:8px;"
            "font-size:13px;border:1px solid #ddd;border-radius:4px}"
            "table.error-table th{background:#f8f9fa;padding:6px 10px;text-align:left;"
            "border-bottom:1px solid #ddd;color:#555}"
            "table.error-table td{padding:6px 10px;border-bottom:1px solid #eee}"
            ".err-type{white-space:nowrap;font-weight:600;color:#e74c3c}"
            "table.detail-table{border-collapse:collapse;font-size:12px;margin-top:4px}"
            "table.detail-table td{padding:2px 8px;border:1px solid #ddd}"
            ".detail-key{font-weight:600;background:#f8f9fa;white-space:nowrap}"
            ".detail-val{color:#555}"
            ".no-error{color:#999;font-style:italic}"
            ".timestamp{color:#999;font-size:13px;margin-top:8px}"
            "</style></head><body>"
            "<h1>数据完整性校验报告</h1>"
            f'<div class="timestamp">生成时间：{data["generated_at"]}</div>'
            '<div class="summary">'
            f'<div class="summary-card total"><div class="num">{summary["total_tables"]}</div>'
            '<div class="label">校验表数</div></div>'
            f'<div class="summary-card pass"><div class="num">{summary["passed"]}</div>'
            '<div class="label">通过</div></div>'
            f'<div class="summary-card fail"><div class="num">{summary["failed"]}</div>'
            '<div class="label">失败</div></div>'
            f'<div class="summary-card err"><div class="num">{summary["total_errors"]}</div>'
            '<div class="label">错误总数</div></div>'
            "</div>"
            '<table class="main-table">'
            "<tr><th>表名</th><th>状态</th><th>错误详情</th></tr>"
            + "".join(rows_html)
            + "</table></body></html>"
        )

    def export_html(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html())
