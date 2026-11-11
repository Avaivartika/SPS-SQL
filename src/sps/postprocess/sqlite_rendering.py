"""Render local edits without translating existing SQLite CAST semantics.

SQLGlot's SQLite generator is a transpiler. For example, it emits DATE(x)
for CAST(x AS DATE), and REAL for NUMERIC. SQLite itself gives those forms
different meanings. Preserve each parsed type spelling while editing other
parts of the query, including nested CASTs and user-defined type names.
"""
from sqlglot import exp
from sqlglot.dialects.sqlite import SQLite
from sqlglot.generator import Generator as SQLGenerator


_TYPE_SOURCE = 'sps_sqlite_type_source'


class NativeSQLite(SQLite):
    """SQLite-to-SQLite AST editing, retaining unchanged CAST target types."""

    class Parser(SQLite.Parser):
        def _parse_types(
            self, check_func: bool = False, schema: bool = False,
            allow_identifiers: bool = True, with_collation: bool = False,
        ) -> exp.Expr | None:
            start = self._curr.start if self._curr is not None else None
            result = super()._parse_types(
                check_func=check_func, schema=schema,
                allow_identifiers=allow_identifiers, with_collation=with_collation,
            )
            if isinstance(result, exp.DataType) and start is not None and self._prev is not None:
                result.meta[_TYPE_SOURCE] = self.sql[start:self._prev.end + 1]
            return result

    class Generator(SQLite.Generator):
        def datatype_sql(self, expression: exp.DataType) -> str:
            source = expression.meta.get(_TYPE_SOURCE)
            return source if source is not None else super().datatype_sql(expression)

        def cast_sql(self, expression: exp.Cast, safe_prefix: str | None = None) -> str:
            target = expression.args.get('to')
            if target is not None and _TYPE_SOURCE in target.meta:
                return SQLGenerator.cast_sql(self, expression, safe_prefix=safe_prefix)
            return super().cast_sql(expression, safe_prefix=safe_prefix)
