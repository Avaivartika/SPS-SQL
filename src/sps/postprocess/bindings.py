"""SQL-scope column bindings used when describing candidate queries.

Resolve physical columns through query-local aliases and derived projections.
This inspects the existing AST without rewriting SQL, expanding answer slots,
or guessing a table when the binding is ambiguous. Recursive dependencies are
bounded by a visited-output set rather than a query-specific depth limit.
"""
from __future__ import annotations

from typing import Any

from sqlglot import exp
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import Scope, build_scope


def _unique(values):
    result, seen = [], set()
    for value in values:
        if value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result


class SQLBindings:
    """Bind columns on the original AST to physical schema names."""

    def __init__(self, root: exp.Expression, schema: dict[str, Any] | None = None):
        # SQLGlot's scope keys are case-sensitive until normalized. Analyze a
        # private SQLite-normalized copy so CTE spelling is not mistaken for
        # a physical table and the caller's SQL/literal text remains intact.
        self.root = root.copy()
        self.analysis_nodes = {id(original): copied for original, copied
                               in zip(root.walk(), self.root.walk(), strict=True)}
        normalize_identifiers(self.root, dialect="sqlite")
        root_scope = build_scope(self.root)
        self.scopes = list(root_scope.traverse()) if root_scope else []
        self.by_expression = {id(scope.expression): scope for scope in self.scopes}
        self.sources = {
            id(scope): {name.casefold(): source for name, (_, source)
                        in scope.selected_sources.items()}
            for scope in self.scopes
        }
        schema = schema or {}
        tables = schema.get("table_names_original") or schema.get("table_names") or []
        columns = schema.get("column_names_original") or schema.get("column_names") or []
        self.table_names = {str(name).casefold(): str(name) for name in tables}
        self.table_columns = {str(name).casefold(): {} for name in tables}
        for index, name in columns:
            if int(index) >= 0 and str(name) != "*":
                self.table_columns[str(tables[int(index)]).casefold()][str(name).casefold()] = str(name)

    def _scope(self, node: exp.Expression) -> Scope | None:
        while node is not None:
            if id(node) in self.by_expression:
                return self.by_expression[id(node)]
            node = node.parent
        return None

    def _physical(self, table: exp.Table, name: str) -> str:
        key = table.name.casefold()
        return (self.table_names.get(key, table.name) + "." +
                self.table_columns.get(key, {}).get(name.casefold(), name))

    def tables(self) -> list[str]:
        """Physical relations, excluding names bound to CTE/derived scopes."""
        result = []
        for table in self.root.find_all(exp.Table):
            scope = self._scope(table)
            source = self.sources.get(id(scope), {}).get(table.alias_or_name.casefold())
            if isinstance(source, exp.Table) and source.name:
                result.append(self.table_names.get(source.name.casefold(), source.name))
        return _unique(result)

    def columns(self, expression: exp.Expression) -> list[str]:
        if id(expression) not in self.analysis_nodes:
            raise ValueError("Column analysis requires a node from the original query AST")
        return self._columns(self.analysis_nodes[id(expression)], frozenset())

    def _columns(self, expression, visited):
        result = []
        for column in expression.find_all(exp.Column):
            scope = self._scope(column)
            values = self._column(column, scope, visited) if scope else None
            if values is None:
                # Preserve unresolved information; do not borrow a matching
                # column from an unrelated table elsewhere in the database.
                values = [f"{column.table}.{column.name}" if column.table else column.name]
            result.extend(values)
        return _unique(result)

    def _alias_visible(self, column, scope, *, order_only=False):
        node = column.parent
        while node is not None and node is not scope.expression:
            if node is scope.expression.args.get("order"):
                return True
            if not order_only and any(node is scope.expression.args.get(key)
                                      for key in ("where", "group", "having")):
                return True
            node = node.parent
        return False

    def _column(self, column, scope, visited):
        name, qualifier = column.name, column.table.casefold()
        sources = self.sources[id(scope)]
        if qualifier:
            source = sources.get(qualifier)
            if source is not None:
                return self._source(source, name, visited)
        else:
            # SQLite ORDER BY resolves output aliases before source columns.
            # WHERE/GROUP BY/HAVING prefer source columns when names collide.
            if self._alias_visible(column, scope, order_only=True):
                value = self._output(scope, name, visited, aliases_only=True)
                if value is not None:
                    return value
            matches = [source for source in sources.values()
                       if self._exposes(source, name)]
            if len(matches) == 1:
                return self._source(matches[0], name, visited)
            if len(matches) > 1:
                return None
            if self._alias_visible(column, scope):
                value = self._output(scope, name, visited, aliases_only=True)
                if value is not None:
                    return value
        if scope.can_be_correlated and scope.parent is not None:
            return self._column(column, scope.parent, visited)
        return None

    def _exposes(self, source, name):
        if isinstance(source, exp.Table):
            known = self.table_columns.get(source.name.casefold())
            return known is None or name.casefold() in known or name == "*"
        return bool(self._output_positions(source, name)) or self._has_star(source)

    def _source(self, source, name, visited):
        if isinstance(source, exp.Table):
            return [self._physical(source, name)] if source.name else None
        return self._output(source, name, visited)

    @staticmethod
    def _has_star(scope):
        return any(projection.is_star for projection in scope.expression.selects)

    @staticmethod
    def _output_positions(scope, name, *, aliases_only=False):
        positions = []
        for index, projection in enumerate(scope.expression.selects):
            if aliases_only and not isinstance(projection, exp.Alias):
                continue
            exposed = (scope.outer_columns[index] if index < len(scope.outer_columns)
                       else projection.alias_or_name)
            if exposed.casefold() == name.casefold():
                positions.append(index)
        return positions

    def _output(self, scope, name, visited, *, aliases_only=False):
        marker = (id(scope), name.casefold())
        if marker in visited:
            return None
        visited = visited | {marker}
        if scope.outer_columns and self._has_star(scope):
            # An explicit column list names expanded positions, not AST
            # projection positions. Retain unresolved lineage here rather
            # than silently assigning a name to the wrong expanded column.
            return None
        positions = self._output_positions(scope, name, aliases_only=aliases_only)
        for index, projection in enumerate(scope.expression.selects):
            if index in positions:
                return self._projection(scope, index, visited)
            if aliases_only or not projection.is_star:
                continue
            available = self.sources[id(scope)]
            if isinstance(projection, exp.Column) and projection.table:
                source = available.get(projection.table.casefold())
                available = {projection.table: source} if source is not None else {}
            matching = [source for source in available.values() if self._exposes(source, name)]
            if matching:
                # The star can precede a duplicate explicitly named output.
                # A unique source is traceable. Multiple sources may involve
                # USING/coalesced joins, so do not guess a physical lineage.
                return self._source(matching[0], name, visited) if len(matching) == 1 else None
        return None

    def _projection(self, scope, index, visited):
        if isinstance(scope.expression, exp.SetOperation):
            if any(self._has_star(branch) for branch in scope.set_operation_scopes):
                return None
            branches = [self._projection(branch, index, visited) for branch in scope.set_operation_scopes]
            return None if any(values is None for values in branches) else _unique(
                value for values in branches for value in values)
        projections = scope.expression.selects
        if index >= len(projections):
            return []
        return self._columns(projections[index], visited)
