-- Add runtime column to parameter_presets table
-- Runtime can be 'sglang', 'vllm', or NULL for universal presets

ALTER TABLE parameter_presets ADD COLUMN runtime VARCHAR(50);

-- Create index on runtime for filtering
CREATE INDEX idx_parameter_presets_runtime ON parameter_presets(runtime);

-- Update existing system presets with runtime values (can be customized)
-- Leaving them NULL for now (universal), can be updated manually if needed
