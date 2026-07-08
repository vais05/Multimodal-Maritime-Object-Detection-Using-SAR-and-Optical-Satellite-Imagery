/*
  # Maritime Surveillance Platform — Initial Schema

  1. New Tables
    - `profiles` — Extended user profile linked to auth.users
      - `id` (uuid, FK to auth.users)
      - `full_name`, `organization`, `role`, `avatar_url`
      - `created_at`, `updated_at`
    - `notebooks` — Uploaded Jupyter notebook files per phase
      - `id`, `user_id`, `phase` (0-8), `filename`, `content` (JSON), `uploaded_at`
    - `detection_history` — Past detection runs
      - `id`, `user_id`, `input_type` (image/video), `filename`, `result_json`, `created_at`
    - `phase_runs` — Records of pipeline phase executions
      - `id`, `user_id`, `phase`, `status`, `output_json`, `started_at`, `finished_at`

  2. Security
    - RLS enabled on all tables
    - Users can only read/write their own data
*/

-- Profiles
CREATE TABLE IF NOT EXISTS profiles (
  id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name text DEFAULT '',
  organization text DEFAULT '',
  role text DEFAULT 'researcher',
  avatar_url text DEFAULT '',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
  ON profiles FOR SELECT
  TO authenticated
  USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
  ON profiles FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON profiles FOR UPDATE
  TO authenticated
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

-- Notebooks
CREATE TABLE IF NOT EXISTS notebooks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  phase integer NOT NULL CHECK (phase >= 0 AND phase <= 8),
  filename text NOT NULL DEFAULT '',
  content jsonb DEFAULT '{}',
  file_size bigint DEFAULT 0,
  uploaded_at timestamptz DEFAULT now()
);

ALTER TABLE notebooks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own notebooks"
  ON notebooks FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own notebooks"
  ON notebooks FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own notebooks"
  ON notebooks FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can update own notebooks"
  ON notebooks FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Detection history
CREATE TABLE IF NOT EXISTS detection_history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  input_type text NOT NULL DEFAULT 'image' CHECK (input_type IN ('image', 'video')),
  filename text NOT NULL DEFAULT '',
  ship_count integer DEFAULT 0,
  confidence_avg numeric(5,4) DEFAULT 0,
  processing_time_ms integer DEFAULT 0,
  result_json jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

ALTER TABLE detection_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own detections"
  ON detection_history FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own detections"
  ON detection_history FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own detections"
  ON detection_history FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Phase runs
CREATE TABLE IF NOT EXISTS phase_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  phase integer NOT NULL CHECK (phase >= 0 AND phase <= 8),
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','running','completed','failed')),
  output_json jsonb DEFAULT '{}',
  error_message text DEFAULT '',
  started_at timestamptz DEFAULT now(),
  finished_at timestamptz
);

ALTER TABLE phase_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own phase runs"
  ON phase_runs FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own phase runs"
  ON phase_runs FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own phase runs"
  ON phase_runs FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Trigger to create profile on signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS trigger AS $$
BEGIN
  INSERT INTO profiles (id, full_name, avatar_url)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data->>'full_name', ''),
    COALESCE(NEW.raw_user_meta_data->>'avatar_url', '')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION handle_new_user();
