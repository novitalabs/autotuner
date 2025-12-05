"""
Generic low-level database tools for agent.

These tools provide low-level database access:
- Read operations are SAFE (no authorization required)
- Write operations REQUIRE authorization (DATABASE_WRITE scope)

For business-level operations, prefer high-level tools:
- task_tools.py for task management
- preset_tools.py for parameter presets
- experiment_tools.py for experiments (future)
"""

from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from web.agent.tools.base import register_tool, ToolCategory, AuthorizationScope
import json
from typing import Optional


@tool
@register_tool(ToolCategory.DATABASE)
async def query_records(
    table_name: str,
    filters: dict = None,
    limit: int = 10,
    db: AsyncSession = None
) -> str:
    """
    Generic low-level query to fetch records from any table.

    Args:
        table_name: Table name (tasks, experiments, parameter_presets, etc.)
        filters: Optional dict of field:value filters, e.g., {"status": "running", "deployment_mode": "docker"}
        limit: Maximum number of records to return (default 10, max 100)

    Returns:
        JSON string with list of records

    Examples:
        - List all tasks: table_name="tasks", filters=None
        - List running tasks: table_name="tasks", filters={"status": "running"}
        - List experiments for task 1: table_name="experiments", filters={"task_id": 1}
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Validate table name
    valid_tables = ["tasks", "experiments", "parameter_presets", "chat_sessions", "chat_messages", "agent_event_subscriptions"]
    if table_name not in valid_tables:
        return json.dumps({
            "error": f"Invalid table name. Must be one of: {', '.join(valid_tables)}"
        })

    limit = min(limit, 100)  # Cap at 100

    try:
        # Build query
        query = f"SELECT * FROM {table_name}"
        params = {}

        if filters:
            where_clauses = []
            for i, (field, value) in enumerate(filters.items()):
                param_name = f"filter_{i}"
                where_clauses.append(f"{field} = :{param_name}")
                params[param_name] = value
            query += " WHERE " + " AND ".join(where_clauses)

        query += f" LIMIT :limit"
        params["limit"] = limit

        result = await db.execute(text(query), params)
        rows = result.fetchall()

        if rows:
            columns = result.keys()
            results = [dict(zip(columns, row)) for row in rows]
            return json.dumps({
                "success": True,
                "table": table_name,
                "row_count": len(results),
                "results": results
            }, indent=2)
        else:
            return json.dumps({
                "success": True,
                "table": table_name,
                "row_count": 0,
                "results": []
            })

    except Exception as e:
        return json.dumps({
            "error": f"Query failed: {str(e)}"
        })


@tool
@register_tool(ToolCategory.DATABASE)
async def get_record_by_id(
    table_name: str,
    record_id: int,
    db: AsyncSession = None
) -> str:
    """
    Generic low-level query to fetch a single record by ID.

    Args:
        table_name: Table name (tasks, experiments, parameter_presets, etc.)
        record_id: Primary key ID of the record

    Returns:
        JSON string with record details

    Examples:
        - Get task 1: table_name="tasks", record_id=1
        - Get experiment 5: table_name="experiments", record_id=5
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Validate table name
    valid_tables = ["tasks", "experiments", "parameter_presets", "chat_sessions", "chat_messages", "agent_event_subscriptions"]
    if table_name not in valid_tables:
        return json.dumps({
            "error": f"Invalid table name. Must be one of: {', '.join(valid_tables)}"
        })

    try:
        query = text(f"SELECT * FROM {table_name} WHERE id = :id")
        result = await db.execute(query, {"id": record_id})
        row = result.fetchone()

        if row:
            columns = result.keys()
            record = dict(zip(columns, row))
            return json.dumps({
                "success": True,
                "table": table_name,
                "record": record
            }, indent=2)
        else:
            return json.dumps({
                "error": f"No record found with id {record_id} in table {table_name}"
            })

    except Exception as e:
        return json.dumps({
            "error": f"Query failed: {str(e)}"
        })


# ============================================================================
# Generic Low-Level Write Operations (Require Authorization)
# ============================================================================

@tool
@register_tool(
    ToolCategory.DATABASE,
    requires_auth=True,
    auth_scope=AuthorizationScope.DATABASE_WRITE
)
async def update_database_field(
    table_name: str,
    record_id: int,
    field_name: str,
    field_value: str,
    db: AsyncSession = None
) -> str:
    """
    Generic low-level operation to update a single field in any database table.

    **REQUIRES AUTHORIZATION**: This is a privileged operation that requires user approval.

    Args:
        table_name: Table name (tasks, experiments, parameter_presets, chat_sessions, etc.)
        record_id: Primary key ID of the record to update
        field_name: Name of the field to update
        field_value: New value for the field (as string, will be converted)

    Returns:
        JSON string with success/error message

    Examples:
        - Update task description: table_name="tasks", record_id=1, field_name="description", field_value="New desc"
        - Update experiment status: table_name="experiments", record_id=5, field_name="status", field_value="failed"
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Validate table name
    valid_tables = ["tasks", "experiments", "parameter_presets", "chat_sessions", "chat_messages", "agent_event_subscriptions"]
    if table_name not in valid_tables:
        return json.dumps({
            "error": f"Invalid table name. Must be one of: {', '.join(valid_tables)}"
        })

    try:
        # Use raw SQL for generic update
        query = text(f"UPDATE {table_name} SET {field_name} = :value WHERE id = :id")
        result = await db.execute(query, {"value": field_value, "id": record_id})
        await db.commit()

        if result.rowcount == 0:
            return json.dumps({
                "error": f"No record found with id {record_id} in table {table_name}"
            })

        return json.dumps({
            "success": True,
            "message": f"Updated {table_name}.{field_name} for record {record_id}",
            "table": table_name,
            "record_id": record_id,
            "field": field_name,
            "new_value": field_value
        }, indent=2)

    except Exception as e:
        await db.rollback()
        return json.dumps({
            "error": f"Failed to update record: {str(e)}"
        })


@tool
@register_tool(
    ToolCategory.DATABASE,
    requires_auth=True,
    auth_scope=AuthorizationScope.DATABASE_WRITE
)
async def execute_raw_sql(
    sql_query: str,
    is_write: bool = False,
    db: AsyncSession = None
) -> str:
    """
    Execute raw SQL query on the database.

    **REQUIRES AUTHORIZATION**: This is a privileged operation that requires user approval.
    **DANGEROUS**: Use with caution. Prefer high-level business tools when available.

    Args:
        sql_query: SQL query to execute (SELECT, UPDATE, DELETE, etc.)
        is_write: Set to True if this is a write operation (INSERT/UPDATE/DELETE)

    Returns:
        JSON string with query results or success message

    Safety Notes:
        - Only allows operations on application tables (tasks, experiments, etc.)
        - Read queries return results as JSON
        - Write queries return affected row count
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Basic safety check: prevent operations on system tables
    dangerous_keywords = ["DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
    query_upper = sql_query.upper()

    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return json.dumps({
                "error": f"Dangerous operation '{keyword}' not allowed. Use high-level tools instead."
            })

    try:
        result = await db.execute(text(sql_query))

        if is_write:
            await db.commit()
            return json.dumps({
                "success": True,
                "message": "Query executed successfully",
                "rows_affected": result.rowcount
            }, indent=2)
        else:
            # Fetch results for SELECT queries
            rows = result.fetchall()
            if rows:
                # Convert rows to list of dicts
                columns = result.keys()
                results = [dict(zip(columns, row)) for row in rows]
                return json.dumps({
                    "success": True,
                    "row_count": len(results),
                    "results": results
                }, indent=2)
            else:
                return json.dumps({
                    "success": True,
                    "row_count": 0,
                    "results": []
                })

    except Exception as e:
        if is_write:
            await db.rollback()
        return json.dumps({
            "error": f"Query execution failed: {str(e)}"
        })
