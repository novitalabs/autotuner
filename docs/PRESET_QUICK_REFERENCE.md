# Parameter Preset System - Quick Reference

## Overview

The Parameter Preset System allows users to save, manage, and reuse parameter configurations for LLM inference tuning tasks. Users can apply multiple presets simultaneously with intelligent merging.

## Key Concepts

### Presets
Reusable parameter configurations stored in the database with metadata:
- **Name**: Unique identifier
- **Category**: Organization (performance, memory, custom)
- **Parameters**: Engine configuration values
- **System Presets**: Built-in, cannot be deleted

### Merge Strategies
When applying multiple presets:
1. **Union** (Default): Combines all values from all presets
2. **Intersection**: Only keeps values present in all presets
3. **Last Wins**: Later presets override earlier ones

## Quick Usage Guide

### For Users

**Creating a Task with Presets:**
1. Go to "New Task" page
2. Enable "Use Parameter Presets"
3. Select one or more presets from dropdown
4. Choose merge strategy (if multiple selected)
5. Preview merged parameters
6. Optionally fine-tune manually
7. Submit task

**Managing Presets:**
1. Navigate to "Presets" page
2. View all presets in table
3. Filter by category
4. Create new preset with "Create" button
5. Edit existing presets (except system presets)
6. Export presets as JSON files
7. Import presets from JSON files

**Example Workflow:**
```
1. Select "High Throughput" preset
   → Parameters: {"tp-size": [2,4], "mem-fraction-static": [0.9]}

2. Add "Memory Efficient" preset
   → Merged (union): {"tp-size": [1,2,4], "mem-fraction-static": [0.7,0.75,0.9]}

3. Switch to "Last Wins" strategy
   → Result: {"tp-size": [1], "mem-fraction-static": [0.7,0.75]}
   → "Memory Efficient" overrides "High Throughput"

4. Fine-tune: Change tp-size to [2]

5. Create task with final parameters
```

### For Developers

**Backend Implementation:**
```bash
# 1. Database model
src/web/db/models.py → Add ParameterPreset class

# 2. API routes
src/web/routes/presets.py → CRUD + merge endpoints

# 3. Merge logic
src/utils/preset_merger.py → PresetMerger class

# 4. System presets
src/web/db/seed_presets.py → Seed on startup
```

**Frontend Implementation:**
```bash
# 1. Services
frontend/src/services/presetService.ts → API client

# 2. Components
frontend/src/components/PresetSelector.tsx → Multi-select + preview

# 3. Pages
frontend/src/pages/Presets.tsx → Management UI
frontend/src/pages/NewTask.tsx → Integration

# 4. Types
frontend/src/types/preset.ts → TypeScript interfaces
```

**Testing:**
```bash
# Backend
pytest tests/test_preset_merger.py
pytest tests/test_preset_routes.py

# Frontend
cd frontend && npm run test

# Manual E2E
./scripts/start_dev.sh  # Backend
cd frontend && npm run dev  # Frontend
# Test via UI at http://localhost:5173
```

## API Reference

### Endpoints

```
GET    /api/presets/           List all presets
GET    /api/presets/{id}       Get preset by ID
POST   /api/presets/           Create new preset
PUT    /api/presets/{id}       Update preset
DELETE /api/presets/{id}       Delete preset
POST   /api/presets/import     Import from JSON
GET    /api/presets/{id}/export Export to JSON
POST   /api/presets/merge      Merge multiple presets
```

### Example API Calls

**Merge presets:**
```bash
curl -X POST http://localhost:8000/api/presets/merge \
  -H "Content-Type: application/json" \
  -d '{
    "preset_ids": [1, 2],
    "merge_strategy": "union"
  }'
```

**Create preset:**
```bash
curl -X POST http://localhost:8000/api/presets/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Custom Preset",
    "category": "custom",
    "parameters": {
      "tp-size": [4, 8],
      "mem-fraction-static": [0.85]
    }
  }'
```

**Export preset:**
```bash
curl http://localhost:8000/api/presets/1/export > preset-high-perf.json
```

**Import preset:**
```bash
curl -X POST http://localhost:8000/api/presets/import \
  -F "file=@preset-high-perf.json"
```

## Merge Strategy Cheatsheet

| Strategy      | Use Case                  | Example Result                      |
|---------------|---------------------------|-------------------------------------|
| Union         | Explore all combinations  | A=[1,2] + B=[3] → [1,2,3]          |
| Intersection  | Conservative tuning       | A=[1,2] + B=[2,3] → [2]            |
| Last Wins     | Base + override           | A=[1,2] + B=[3] → [3] (B overrides)|

## System Presets

Built-in presets available by default:

| Name             | Category    | Parameters                                    | Use Case              |
|------------------|-------------|-----------------------------------------------|-----------------------|
| Memory Efficient | memory      | tp-size=[1], mem-fraction-static=[0.7,0.75]  | Small GPU, low memory |
| High Throughput  | performance | tp-size=[2,4], mem-fraction-static=[0.9]     | Max tokens/second     |
| Low Latency      | performance | tp-size=[1,2], schedule-policy=["lpm"]       | Min latency           |
| Balanced         | general     | tp-size=[1,2], mem-fraction-static=[0.85]    | General use           |

## File Format (Import/Export)

```json
{
  "version": "1.0",
  "preset": {
    "name": "High Performance SGLang",
    "description": "Optimized for high throughput with SGLang runtime",
    "category": "performance",
    "parameters": {
      "tp-size": [2, 4],
      "mem-fraction-static": [0.9],
      "schedule-policy": ["fcfs"]
    },
    "metadata": {
      "author": "system",
      "tags": ["sglang", "throughput", "production"],
      "recommended_for": ["llama-3-70b", "llama-3-405b"]
    }
  }
}
```

## Database Schema

```sql
CREATE TABLE parameter_presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(100),
    is_system BOOLEAN DEFAULT FALSE,
    parameters JSON NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Common Issues

### "Preset not found"
- Check preset ID exists: `SELECT * FROM parameter_presets WHERE id=?`
- Verify database connection

### "Invalid merge strategy"
- Must be one of: `union`, `intersection`, `last_wins`
- Check spelling and case

### "Cannot delete system preset"
- System presets (`is_system=true`) are protected
- Create a custom preset instead

### "Import validation failed"
- Ensure JSON has `version` and `preset` fields
- Parameters must be arrays: `{"tp-size": [1,2]}` not `{"tp-size": 1}`

### "No common values in intersection"
- Intersection of `[1,2]` and `[3,4]` is empty
- Switch to union or last_wins strategy

## Performance Tips

1. **Use system presets as starting point** instead of creating from scratch
2. **Union strategy** for initial exploration, then narrow down
3. **Export frequently used presets** for backup and sharing
4. **Categorize presets** for easier organization
5. **Add metadata tags** for searchability

## Future Enhancements

- [ ] Preset versioning and change tracking
- [ ] Preset sharing between users/teams
- [ ] AI-suggested presets based on model/hardware
- [ ] Preset performance analytics
- [ ] Smart merge with conflict resolution
- [ ] Preset dependencies (requires other presets)
- [ ] Validation rules for parameter combinations

## Documentation

- **Full Design**: `docs/PRESET_SYSTEM_DESIGN.md`
- **Implementation Guide**: `docs/PRESET_IMPLEMENTATION_GUIDE.md`
- **Architecture Diagrams**: `docs/PRESET_ARCHITECTURE_DIAGRAM.md`
- **Quick Reference**: `docs/PRESET_QUICK_REFERENCE.md` (this file)

## Support

For issues or questions:
1. Check `docs/TROUBLESHOOTING.md`
2. Review implementation guide for debugging steps
3. Check backend logs: `logs/worker.log`
4. Check frontend console: Browser DevTools
5. Verify database state: `sqlite3 ~/.local/share/inference-autotuner/autotuner.db`

---

## System Architecture

### Database Schema

Presets stored in `parameter_presets` table:
- `id`, `name`, `description`, `category`
- `is_system` (boolean) - System presets cannot be deleted
- `parameters` (JSON) - Parameter configuration
- `metadata` (JSON) - Additional info (author, tags)

### API Endpoints

- `GET /api/presets/` - List all presets
- `GET /api/presets/{id}` - Get preset details
- `POST /api/presets/` - Create new preset
- `PUT /api/presets/{id}` - Update preset
- `DELETE /api/presets/{id}` - Delete preset (non-system only)
- `POST /api/presets/import` - Import from JSON
- `GET /api/presets/{id}/export` - Export to JSON

### Merge Strategies

When applying multiple presets to a task:

1. **Override** (default): Last preset wins
2. **Append**: Combine arrays, unique values
3. **Deep Merge**: Recursively merge objects

### Built-in System Presets

- `high-throughput` - Maximize throughput
- `low-latency` - Minimize latency
- `memory-efficient` - Reduce memory usage
- `balanced` - Balance performance and resources

System presets are created on first startup and cannot be deleted.

