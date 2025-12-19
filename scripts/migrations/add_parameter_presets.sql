-- Migration: Add parameter_presets table
-- Date: 2025-10-30

CREATE TABLE IF NOT EXISTS parameter_presets (
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

CREATE INDEX IF NOT EXISTS idx_preset_category ON parameter_presets(category);
CREATE INDEX IF NOT EXISTS idx_preset_system ON parameter_presets(is_system);
CREATE INDEX IF NOT EXISTS idx_preset_name ON parameter_presets(name);
