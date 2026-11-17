from __future__ import annotations

import sqlite3
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class ProjectError(Exception):
    def __init__(self, status, code, message, fields=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.fields = fields or {}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def validate_project(values, label="项目"):
    fields = {}
    name = values.get("name")
    description = values.get("description", "")
    if not isinstance(name, str):
        fields["name"] = f"请输入{label}名称。"
        name = ""
    name = name.strip()
    if not 1 <= len(name) <= 60 or any(unicodedata.category(c) == "Cc" for c in name):
        fields["name"] = f"{label}名称需为 1–60 个字符，不能包含换行或控制字符。"
    if not isinstance(description, str):
        fields["description"] = f"{label}说明需为文本。"
        description = ""
    description = description.strip()
    if len(description) > 1000 or "\x00" in description:
        fields["description"] = f"{label}说明最多 1000 个字符，不能包含空字符。"
    if fields:
        code = {"项目": "INVALID_PROJECT", "数据集": "INVALID_DATASET", "任务": "INVALID_TASK"}[label]
        raise ProjectError(422, code, f"请检查{label}内容。", fields)
    return name, description, unicodedata.normalize("NFKC", name).casefold()


def validate_revision(value, label="项目"):
    if type(value) is not int or value < 1:
        raise ProjectError(422, "INVALID_REVISION", f"{label}信息已过期，请刷新后重试。")
    return value


class Storage:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            self.migrate(db)

    def migrate(self, db):
        existing=db.execute("SELECT 1 FROM sqlite_master WHERE name='schema_version'").fetchone()
        if existing and db.execute('SELECT version FROM schema_version').fetchone()[0]in (14,15):
            backup_dir=self.path.parent/'backups';backup_dir.mkdir(exist_ok=True)
            backup=sqlite3.connect(backup_dir/f'schema-v{db.execute("SELECT version FROM schema_version").fetchone()[0]}-{uuid.uuid4().hex}.db')
            try:db.backup(backup)
            finally:backup.close()
        self.migrate_base(db)
        version=db.execute('SELECT version FROM schema_version').fetchone()[0]
        if version>=16:return
        if not db.in_transaction:db.execute('BEGIN IMMEDIATE')
        if version<15:self.create_segment_tables(db)
        self.create_roi_tables(db)
        db.execute('UPDATE schema_version SET version=16')

    @staticmethod
    def create_roi_tables(db):
        db.execute('CREATE TABLE IF NOT EXISTS dataset_rois (dataset_id TEXT PRIMARY KEY REFERENCES datasets(id) ON DELETE CASCADE, region_json TEXT NOT NULL, revision INTEGER NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS roi_annotations (task_id TEXT NOT NULL,image_id TEXT NOT NULL,roi_key TEXT NOT NULL,state TEXT NOT NULL,revision INTEGER NOT NULL,boxes_json TEXT NOT NULL,PRIMARY KEY(task_id,image_id,roi_key),FOREIGN KEY(task_id,image_id) REFERENCES task_images(task_id,image_id) ON DELETE CASCADE)')

    @staticmethod
    def create_segment_tables(db):
        db.execute('''CREATE TABLE segmentation_polygons (
            task_id TEXT NOT NULL, image_id TEXT NOT NULL, position INTEGER NOT NULL,
            class_id TEXT NOT NULL, points_json TEXT NOT NULL,
            PRIMARY KEY(task_id,image_id,position),
            FOREIGN KEY(task_id,image_id) REFERENCES task_images(task_id,image_id) ON DELETE CASCADE,
            FOREIGN KEY(task_id,class_id) REFERENCES task_classes(task_id,id) ON DELETE RESTRICT)''')

    def migrate_base(self, db):
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT IN ('dataset_rois','roi_annotations')"
        )}
        if not tables:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
            self.create_project_table(db, "projects")
            self.create_dataset_table(db)
            self.create_image_table(db)
            self.create_task_tables(db)
            self.create_annotation_tables(db)
            self.create_preparation_tables(db)
            self.create_version_tables(db)
            self.create_training_tables(db)
            self.create_model_tables(db)
            self.create_training_visibility(db)
            self.create_evaluation_tables(db)
            self.create_inference_tables(db)
            db.execute("INSERT INTO schema_version VALUES (14)")
            return
        if "schema_version" not in tables:
            raise ProjectError(503, "SCHEMA_UNSUPPORTED", "数据库格式无法识别，请先核对数据版本。")
        versions = [row[0] for row in db.execute("SELECT version FROM schema_version")]
        if versions in ([14],[15],[16]):
            return
        if versions == [13]:
            backup_dir=self.path.parent/'backups';backup_dir.mkdir(exist_ok=True)
            backup=sqlite3.connect(backup_dir/f'schema-v13-{uuid.uuid4().hex}.db')
            try:db.backup(backup)
            finally:backup.close()
            db.execute('BEGIN IMMEDIATE')
            self.create_inference_tables(db)
            db.execute('UPDATE schema_version SET version=14')
            return
        if versions == [12]:
            backup_dir=self.path.parent/'backups';backup_dir.mkdir(exist_ok=True)
            backup=sqlite3.connect(backup_dir/f'schema-v12-{uuid.uuid4().hex}.db')
            try:db.backup(backup)
            finally:backup.close()
            db.execute('BEGIN IMMEDIATE')
            self.create_evaluation_tables(db)
            self.create_inference_tables(db)
            db.execute('UPDATE schema_version SET version=14')
            return
        if versions == [11]:
            backup_dir=self.path.parent/'backups';backup_dir.mkdir(exist_ok=True)
            backup=sqlite3.connect(backup_dir/f'schema-v11-{uuid.uuid4().hex}.db')
            try:db.backup(backup)
            finally:backup.close()
            db.execute('BEGIN IMMEDIATE')
            self.create_training_visibility(db)
            self.create_evaluation_tables(db)
            self.create_inference_tables(db)
            db.execute('UPDATE schema_version SET version=14')
            return
        if versions == [10]:
            backup_dir=self.path.parent/'backups';backup_dir.mkdir(exist_ok=True)
            backup=sqlite3.connect(backup_dir/f'schema-v10-{uuid.uuid4().hex}.db')
            try:db.backup(backup)
            finally:backup.close()
            db.execute('BEGIN IMMEDIATE')
            self.create_model_tables(db)
            self.create_training_visibility(db)
            self.create_evaluation_tables(db)
            self.create_inference_tables(db)
            db.execute('UPDATE schema_version SET version=14')
            return
        if versions == [9]:
            backup_dir=self.path.parent/'backups';backup_dir.mkdir(exist_ok=True)
            backup=sqlite3.connect(backup_dir/f'schema-v9-{uuid.uuid4().hex}.db')
            try:db.backup(backup)
            finally:backup.close()
            db.execute('BEGIN IMMEDIATE')
            self.upgrade_training_v10(db)
            self.create_model_tables(db)
            self.create_training_visibility(db)
            self.create_evaluation_tables(db)
            self.create_inference_tables(db)
            db.execute('UPDATE schema_version SET version=14')
            return
        if versions == [8]:
            backup_dir=self.path.parent/'backups';backup_dir.mkdir(exist_ok=True)
            backup=sqlite3.connect(backup_dir/f'schema-v8-{uuid.uuid4().hex}.db')
            try:db.backup(backup)
            finally:backup.close()
            db.execute('BEGIN IMMEDIATE')
            self.create_training_tables(db)
            self.create_model_tables(db)
            self.create_training_visibility(db)
            self.create_evaluation_tables(db)
            self.create_inference_tables(db)
            db.execute('UPDATE schema_version SET version=14')
            return
        expected = {"projects", "schema_version"}
        if versions in ([3], [4]): expected.add('datasets')
        if versions == [4]: expected.add('images')
        if versions == [5]: expected.update({'datasets', 'images', 'vision_tasks', 'task_images'})
        if versions == [6]: expected.update({'datasets','images','vision_tasks','task_images','task_classes','task_annotations','detection_boxes'})
        if versions == [7]: expected.update({'datasets','images','vision_tasks','task_images','task_classes','task_annotations','detection_boxes','task_splits','task_split_images'})
        if versions not in ([1], [2], [3], [4], [5], [6], [7]) or tables != expected:
            raise ProjectError(503, "SCHEMA_UNSUPPORTED", "当前数据库需要专用迁移，请保留数据并核对版本。")

        # Back up the current format before any schema changes.
        backup_dir = self.path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f"schema-v{versions[0]}-{uuid.uuid4().hex}.db"
        backup = sqlite3.connect(backup_path)
        try:
            db.backup(backup)
        finally:
            backup.close()
        db.execute("BEGIN IMMEDIATE")
        if versions == [1]:
            self.create_project_table(db, "projects_v2")
            db.execute(
                "INSERT INTO projects_v2(id,name,name_key,description,revision,created_at,updated_at) "
                "SELECT id,name,name_key,description,revision,created_at,updated_at FROM projects"
            )
            db.execute("DROP TABLE projects")
            db.execute("ALTER TABLE projects_v2 RENAME TO projects")
        if versions in ([1], [2]):
            self.create_dataset_table(db)
        if versions in ([1], [2], [3]):
            self.create_image_table(db)
        if versions not in ([5], [6], [7]):
            self.create_task_tables(db)
        if versions not in ([6], [7]):
            self.create_annotation_tables(db)
        if versions != [7]: self.create_preparation_tables(db)
        self.create_version_tables(db)
        self.create_training_tables(db)
        self.create_model_tables(db)
        self.create_training_visibility(db)
        self.create_evaluation_tables(db)
        self.create_inference_tables(db)
        db.execute("DELETE FROM schema_version")
        db.execute("INSERT INTO schema_version VALUES (14)")

    @staticmethod
    def create_inference_tables(db):
        image_sql=db.execute("SELECT sql FROM sqlite_master WHERE name='images'").fetchone()[0]
        db.execute(image_sql.replace('CREATE TABLE images','CREATE TABLE inference_inputs').replace('dataset_id','task_id').replace('REFERENCES datasets','REFERENCES vision_tasks'))
        db.execute('CREATE INDEX inference_inputs_task_idx ON inference_inputs(task_id)')
        db.execute("""CREATE TABLE inference_upload_failures (
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES vision_tasks(id) ON DELETE RESTRICT,
            filename TEXT NOT NULL, original_sha256 TEXT NOT NULL, byte_size INTEGER NOT NULL,
            error_code TEXT NOT NULL, error TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(task_id,filename,original_sha256))""")
        objects=list(db.execute("SELECT type,name,sql FROM sqlite_master WHERE tbl_name IN ('evaluation_jobs','evaluation_dispatches','evaluation_remote_sessions') AND sql IS NOT NULL ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END"))
        for kind,name,sql in objects:
            if name.endswith('_shared_compute_guard'):continue
            db.execute(sql.replace('evaluation_','inference_'))
        db.execute("""CREATE TABLE inference_job_inputs (
            job_id TEXT NOT NULL REFERENCES inference_jobs(id) ON DELETE RESTRICT,
            position INTEGER NOT NULL CHECK(position>=0),
            image_id TEXT REFERENCES inference_inputs(id) ON DELETE RESTRICT,
            failure_id TEXT REFERENCES inference_upload_failures(id) ON DELETE RESTRICT,
            PRIMARY KEY(job_id,position), UNIQUE(job_id,image_id), UNIQUE(job_id,failure_id),
            CHECK((image_id IS NULL)!=(failure_id IS NULL)))""")
        for table in ('inference_inputs','inference_upload_failures','inference_job_inputs'):
            for action in ('UPDATE','DELETE'):
                db.execute(f"CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'Inference inputs are immutable'); END")
        kinds=('training','evaluation','inference')
        for kind in kinds:
            db.execute(f'DROP TRIGGER IF EXISTS {kind}_shared_compute_guard')
            predicates=' OR '.join(f"EXISTS(SELECT 1 FROM {other}_dispatches WHERE target_id=NEW.target_id AND state='AWAITING_RESULT')" for other in kinds if other!=kind)
            db.execute(f"CREATE TRIGGER {kind}_shared_compute_guard BEFORE INSERT ON {kind}_dispatches WHEN NEW.state='AWAITING_RESULT' AND ({predicates}) BEGIN SELECT RAISE(ABORT,'Compute target is reserved by another task kind'); END")

    @staticmethod
    def create_evaluation_tables(db):
        # Reuse the proven state machine; evaluation history remains a separate object.
        db.execute('CREATE UNIQUE INDEX IF NOT EXISTS model_task_identity ON model_versions(task_id,id)')
        objects=list(db.execute("SELECT type,name,sql FROM sqlite_master WHERE tbl_name IN ('training_jobs','training_dispatches','training_remote_sessions') AND sql IS NOT NULL ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END"))
        for kind,name,sql in objects:
            copied=sql.replace('training_','evaluation_')
            if name.startswith('dispatch_'):copied=copied.replace(name,'evaluation_'+name,1)
            if kind=='table' and name=='training_jobs':
                copied=copied.replace('version_id TEXT NOT NULL,', 'version_id TEXT NOT NULL, model_id TEXT NOT NULL,')
                copied=copied.replace('UNIQUE(task_id,idempotency_key),','UNIQUE(task_id,idempotency_key), FOREIGN KEY(task_id,model_id) REFERENCES model_versions(task_id,id) ON DELETE RESTRICT,')
            if name=='training_job_snapshot_immutable':
                copied=copied.replace('WHEN ', 'WHEN NEW.model_id IS NOT OLD.model_id OR ',1)
            db.execute(copied)
        for first,other in (('training','evaluation'),('evaluation','training')):
            db.execute(f"""CREATE TRIGGER {first}_shared_compute_guard BEFORE INSERT ON {first}_dispatches
                WHEN NEW.state='AWAITING_RESULT' AND EXISTS(SELECT 1 FROM {other}_dispatches WHERE target_id=NEW.target_id AND state='AWAITING_RESULT')
                BEGIN SELECT RAISE(ABORT,'Compute target is reserved by another task kind'); END""")

    @staticmethod
    def create_training_visibility(db):
        db.execute('''CREATE TABLE training_record_visibility (
            job_id TEXT PRIMARY KEY REFERENCES training_jobs(id) ON DELETE RESTRICT,
            deleted_at TEXT, revision INTEGER NOT NULL CHECK(revision>0), updated_at TEXT NOT NULL)''')
        for operation in ('INSERT','UPDATE'):
            db.execute(f'''CREATE TRIGGER training_visibility_active_{operation.lower()} BEFORE {operation} ON training_record_visibility
                WHEN NEW.deleted_at IS NOT NULL AND EXISTS(SELECT 1 FROM training_jobs WHERE id=NEW.job_id AND state IN ('QUEUED','DISPATCHED','RUNNING'))
                BEGIN SELECT RAISE(ABORT,'Active training cannot be hidden'); END''')

    @staticmethod
    def create_model_tables(db):
        db.execute('CREATE UNIQUE INDEX IF NOT EXISTS training_task_identity ON training_jobs(task_id,id)')
        db.execute('''CREATE TABLE model_versions (
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES vision_tasks(id) ON DELETE RESTRICT,
            number INTEGER NOT NULL CHECK(number>0), training_run_id TEXT NOT NULL UNIQUE,
            task_data_version_id TEXT NOT NULL, created_at TEXT NOT NULL,
            manifest_json TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
            UNIQUE(task_id,number),
            FOREIGN KEY(task_id,training_run_id) REFERENCES training_jobs(task_id,id) ON DELETE RESTRICT,
            FOREIGN KEY(task_id,task_data_version_id) REFERENCES task_data_versions(task_id,id) ON DELETE RESTRICT)''')
        db.execute('''CREATE TABLE model_events (
            id TEXT PRIMARY KEY, model_version_id TEXT NOT NULL UNIQUE REFERENCES model_versions(id) ON DELETE RESTRICT,
            event_type TEXT NOT NULL CHECK(event_type='MODEL_REGISTERED'), payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL)''')
        for table in ('model_versions','model_events'):
            for action in ('UPDATE','DELETE'):
                db.execute(f'''CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table}
                    BEGIN SELECT RAISE(ABORT,'Model history must be retained unchanged'); END''')

    def upgrade_training_v10(self,db):
        db.execute('DROP TRIGGER training_job_snapshot_immutable')
        db.execute('DROP TRIGGER training_job_no_delete')
        db.execute('DROP INDEX training_jobs_queue')
        db.execute('ALTER TABLE training_jobs RENAME TO training_jobs_v9')
        self.create_training_tables(db)
        db.execute('INSERT INTO training_jobs SELECT * FROM training_jobs_v9')
        db.execute('DROP TABLE training_jobs_v9')

    @staticmethod
    def create_training_tables(db):
        db.execute('CREATE UNIQUE INDEX IF NOT EXISTS version_task_identity ON task_data_versions(task_id,id)')
        db.execute('''CREATE TABLE training_jobs (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL REFERENCES vision_tasks(id) ON DELETE RESTRICT,
            version_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_sha256 TEXT NOT NULL,
            name TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('QUEUED','CANCELLED','DISPATCHED','RUNNING','SUCCEEDED','FAILED','INTERRUPTED')),
            snapshot_json TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(task_id,idempotency_key),
            FOREIGN KEY(task_id,version_id) REFERENCES task_data_versions(task_id,id) ON DELETE RESTRICT)''')
        db.execute('CREATE INDEX training_jobs_queue ON training_jobs(state,sequence)')
        db.execute('''CREATE TRIGGER training_job_snapshot_immutable BEFORE UPDATE ON training_jobs
            WHEN NEW.sequence IS NOT OLD.sequence OR NEW.id IS NOT OLD.id OR NEW.task_id IS NOT OLD.task_id
            OR NEW.version_id IS NOT OLD.version_id OR NEW.idempotency_key IS NOT OLD.idempotency_key
            OR NEW.request_sha256 IS NOT OLD.request_sha256 OR NEW.name IS NOT OLD.name
            OR NEW.snapshot_json IS NOT OLD.snapshot_json OR NEW.snapshot_sha256 IS NOT OLD.snapshot_sha256
            OR NEW.created_at IS NOT OLD.created_at
            OR NOT ((OLD.state='QUEUED' AND NEW.state IN ('CANCELLED','DISPATCHED'))
                OR (OLD.state='DISPATCHED' AND NEW.state IN ('RUNNING','SUCCEEDED','FAILED','INTERRUPTED'))
                OR (OLD.state='RUNNING' AND NEW.state IN ('SUCCEEDED','FAILED','INTERRUPTED')))
            BEGIN SELECT RAISE(ABORT,'Training snapshots and terminal states are immutable'); END''')
        db.execute('''CREATE TRIGGER training_job_no_delete BEFORE DELETE ON training_jobs
            BEGIN SELECT RAISE(ABORT,'Training records must be retained'); END''')
        db.execute('''CREATE TABLE training_dispatches (
            job_id TEXT PRIMARY KEY REFERENCES training_jobs(id) ON DELETE RESTRICT,
            id TEXT NOT NULL UNIQUE, target_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('AWAITING_RESULT','SUCCEEDED','FAILED','INTERRUPTED')),
            request_json TEXT NOT NULL, package_sha256 TEXT NOT NULL,
            result_sha256 TEXT, result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
        db.execute("CREATE UNIQUE INDEX training_one_target_reservation ON training_dispatches(target_id) WHERE state='AWAITING_RESULT'")
        db.execute('''CREATE TRIGGER dispatch_immutable BEFORE UPDATE ON training_dispatches
            WHEN NEW.job_id IS NOT OLD.job_id OR NEW.id IS NOT OLD.id OR NEW.target_id IS NOT OLD.target_id
            OR NEW.request_json IS NOT OLD.request_json OR NEW.package_sha256 IS NOT OLD.package_sha256
            OR NEW.created_at IS NOT OLD.created_at OR OLD.state!='AWAITING_RESULT'
            OR NEW.state NOT IN ('SUCCEEDED','FAILED','INTERRUPTED') OR NEW.result_json IS NULL OR NEW.result_sha256 IS NULL
            BEGIN SELECT RAISE(ABORT,'Dispatch identity and final results are immutable'); END''')
        db.execute('''CREATE TRIGGER dispatch_no_delete BEFORE DELETE ON training_dispatches
            BEGIN SELECT RAISE(ABORT,'Dispatch records must be retained'); END''')
        db.execute('''CREATE TABLE training_remote_sessions (
            job_id TEXT PRIMARY KEY REFERENCES training_jobs(id) ON DELETE RESTRICT,
            endpoint TEXT NOT NULL, phase TEXT NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0,
            launched INTEGER NOT NULL DEFAULT 0,
            progress_json TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')

    @staticmethod
    def create_version_tables(db):
        db.execute('''CREATE TABLE task_data_versions (
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES vision_tasks(id) ON DELETE RESTRICT,
            number INTEGER NOT NULL CHECK(number>0), fingerprint TEXT NOT NULL, split_revision INTEGER NOT NULL,
            created_at TEXT NOT NULL, manifest_json TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
            UNIQUE(task_id,number), UNIQUE(task_id,fingerprint,split_revision))''')
        db.execute('''CREATE TABLE task_data_version_images (
            version_id TEXT NOT NULL REFERENCES task_data_versions(id) ON DELETE RESTRICT,
            image_id TEXT NOT NULL REFERENCES images(id) ON DELETE RESTRICT,
            PRIMARY KEY(version_id,image_id))''')
        for table in ('task_data_versions','task_data_version_images'):
            for action in ('UPDATE','DELETE'):
                db.execute(f'''CREATE TRIGGER {table}_immutable_{action.lower()} BEFORE {action} ON {table}
                    BEGIN SELECT RAISE(ABORT,'Published data versions are immutable'); END''')

    @staticmethod
    def create_preparation_tables(db):
        db.execute('''CREATE TABLE task_splits (
            task_id TEXT PRIMARY KEY REFERENCES vision_tasks(id) ON DELETE RESTRICT,
            revision INTEGER NOT NULL CHECK(revision>0), fingerprint TEXT NOT NULL,
            seed INTEGER NOT NULL, train_percent INTEGER NOT NULL, val_percent INTEGER NOT NULL,
            test_percent INTEGER NOT NULL, algorithm TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
        db.execute('''CREATE TABLE task_split_images (
            task_id TEXT NOT NULL REFERENCES task_splits(task_id) ON DELETE RESTRICT,
            image_id TEXT NOT NULL, subset TEXT NOT NULL CHECK(subset IN ('train','val','test')),
            PRIMARY KEY(task_id,image_id),
            FOREIGN KEY(task_id,image_id) REFERENCES task_images(task_id,image_id) ON DELETE RESTRICT)''')

    @staticmethod
    def create_annotation_tables(db):
        db.execute('''CREATE TABLE task_classes (
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES vision_tasks(id) ON DELETE RESTRICT,
            name TEXT NOT NULL, name_key TEXT NOT NULL, color TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1 CHECK(revision>0),
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(task_id,name_key), UNIQUE(task_id,id))''')
        db.execute('''CREATE TABLE task_annotations (
            task_id TEXT NOT NULL, image_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('unannotated','annotated','no_target','excluded')),
            revision INTEGER NOT NULL CHECK(revision>0), updated_at TEXT NOT NULL,
            PRIMARY KEY(task_id,image_id),
            FOREIGN KEY(task_id,image_id) REFERENCES task_images(task_id,image_id) ON DELETE RESTRICT)''')
        db.execute('''CREATE TABLE detection_boxes (
            task_id TEXT NOT NULL, image_id TEXT NOT NULL, position INTEGER NOT NULL,
            class_id TEXT NOT NULL, x REAL NOT NULL CHECK(x>=0), y REAL NOT NULL CHECK(y>=0),
            width REAL NOT NULL CHECK(width>0), height REAL NOT NULL CHECK(height>0),
            PRIMARY KEY(task_id,image_id,position),
            FOREIGN KEY(task_id,image_id) REFERENCES task_annotations(task_id,image_id) ON DELETE RESTRICT,
            FOREIGN KEY(task_id,class_id) REFERENCES task_classes(task_id,id) ON DELETE RESTRICT)''')
        db.execute('CREATE INDEX boxes_class_idx ON detection_boxes(task_id,class_id)')

    @staticmethod
    def create_task_tables(db):
        db.execute('CREATE UNIQUE INDEX datasets_project_identity ON datasets(project_id,id)')
        db.execute('CREATE UNIQUE INDEX images_dataset_identity ON images(id,dataset_id)')
        db.execute("""
            CREATE TABLE vision_tasks (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
                dataset_id TEXT NOT NULL,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                task_type TEXT NOT NULL CHECK(task_type IN ('detect','segment')),
                revision INTEGER NOT NULL DEFAULT 1 CHECK(revision>0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_synced_at TEXT NOT NULL,
                UNIQUE(project_id,name_key),
                UNIQUE(id,dataset_id),
                FOREIGN KEY(project_id,dataset_id) REFERENCES datasets(project_id,id) ON DELETE RESTRICT
            )
        """)
        db.execute('CREATE INDEX vision_tasks_dataset_idx ON vision_tasks(dataset_id)')
        db.execute("""
            CREATE TABLE task_images (
                task_id TEXT NOT NULL,
                image_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY(task_id,image_id),
                FOREIGN KEY(task_id,dataset_id) REFERENCES vision_tasks(id,dataset_id) ON DELETE RESTRICT,
                FOREIGN KEY(image_id,dataset_id) REFERENCES images(id,dataset_id) ON DELETE RESTRICT
            )
        """)

    @staticmethod
    def create_image_table(db):
        db.execute("""
            CREATE TABLE images (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
                filename TEXT NOT NULL,
                original_sha256 TEXT NOT NULL,
                display_sha256 TEXT NOT NULL,
                byte_size INTEGER NOT NULL CHECK(byte_size>0),
                width INTEGER NOT NULL CHECK(width>0),
                height INTEGER NOT NULL CHECK(height>0),
                source_width INTEGER NOT NULL,
                source_height INTEGER NOT NULL,
                format TEXT NOT NULL CHECK(format IN ('JPEG','PNG')),
                group_id TEXT NOT NULL DEFAULT '',
                revision INTEGER NOT NULL DEFAULT 1 CHECK(revision>0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(dataset_id,original_sha256)
            )
        """)
        db.execute('CREATE INDEX images_dataset_id_idx ON images(dataset_id)')

    @staticmethod
    def create_dataset_table(db):
        db.execute("""
            CREATE TABLE datasets (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                revision INTEGER NOT NULL DEFAULT 1 CHECK(revision > 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_id, name_key)
            )
        """)
        db.execute("CREATE INDEX datasets_project_id_idx ON datasets(project_id)")

    @staticmethod
    def create_project_table(db, table_name):
        if table_name not in {"projects", "projects_v2"}:
            raise ValueError("Unsupported internal table name")
        db.execute(f"""
            CREATE TABLE {table_name} (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                revision INTEGER NOT NULL DEFAULT 1 CHECK(revision > 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def public(row):
        result = {key: row[key] for key in (
            "id", "name", "description", "revision", "created_at", "updated_at"
        )}
        if "dataset_count" in row.keys():
            result["dataset_count"] = row["dataset_count"]
        if "project_id" in row.keys():
            result["project_id"] = row["project_id"]
        if "image_count" in row.keys():
            result["image_count"] = row["image_count"]
        if "task_count" in row.keys():
            result["task_count"] = row["task_count"]
        if 'cover_image_id' in row.keys() and row['cover_image_id']:result['cover_image_id']=row['cover_image_id']
        return result

    def list_projects(self):
        with self.connect() as db:
            return [self.public(row) for row in db.execute(
                "SELECT projects.*, (SELECT count(*) FROM datasets WHERE project_id=projects.id) AS dataset_count, "
                "(SELECT count(*) FROM vision_tasks WHERE project_id=projects.id) AS task_count "
                "FROM projects ORDER BY updated_at DESC, id DESC"
            )]

    def get(self, project_id, db=None):
        if db is None:
            with self.connect() as connection:
                return self.get(project_id, connection)
        row = db.execute(
            "SELECT projects.*, (SELECT count(*) FROM datasets WHERE project_id=projects.id) AS dataset_count, "
            "(SELECT count(*) FROM vision_tasks WHERE project_id=projects.id) AS task_count "
            "FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row is None:
            raise ProjectError(404, "PROJECT_NOT_FOUND", "项目不存在或已被删除，请返回项目列表。")
        return self.public(row)

    def create(self, values):
        name, description, name_key = validate_project(values)
        project_id, now = uuid.uuid4().hex, timestamp()
        try:
            with self.connect() as db:
                db.execute(
                    "INSERT INTO projects(id,name,name_key,description,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?)", (project_id, name, name_key, description, now, now)
                )
                return self.get(project_id, db)
        except sqlite3.IntegrityError as exc:
            if "projects.name_key" not in str(exc):
                raise
            raise ProjectError(409, "NAME_EXISTS", "已有同名项目，请换一个名称。", {
                "name": "项目名称已被使用（不区分大小写及全角/半角）。"
            }) from exc

    def update(self, project_id, values):
        name, description, name_key = validate_project(values)
        revision = validate_revision(values.get("revision"))
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = self.get(project_id, db)
                if current["revision"] != revision:
                    raise ProjectError(409, "REVISION_CONFLICT", "项目已在其他页面更新。请保留当前输入，关闭编辑并刷新项目后重试。")
                db.execute(
                    "UPDATE projects SET name=?,name_key=?,description=?,revision=revision+1,updated_at=? "
                    "WHERE id=? AND revision=?", (name, name_key, description, timestamp(), project_id, revision)
                )
                return self.get(project_id, db)
        except sqlite3.IntegrityError as exc:
            if "projects.name_key" not in str(exc):
                raise
            raise ProjectError(409, "NAME_EXISTS", "已有同名项目，请换一个名称。", {
                "name": "项目名称已被使用。"
            }) from exc

    def delete(self, project_id, values):
        revision = validate_revision(values.get("revision"))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            project = self.get(project_id, db)
            if project["revision"] != revision:
                raise ProjectError(409, "REVISION_CONFLICT", "项目信息已变化，请关闭弹窗并刷新后重新确认。")
            if values.get("confirmation_name") != project["name"]:
                raise ProjectError(422, "CONFIRMATION_REQUIRED", "请输入完整项目名称以确认删除。", {
                    "confirmation_name": "输入内容与项目名称不一致。"
                })
            if project["dataset_count"]:
                raise ProjectError(409, "PROJECT_NOT_EMPTY", f"项目包含 {project['dataset_count']} 个数据集，请先处理这些数据集后再删除项目。")
            db.execute("DELETE FROM projects WHERE id=?", (project_id,))
            return project

    def list_datasets(self, project_id):
        with self.connect() as db:
            self.get(project_id, db)
            return [self.public(row) for row in db.execute(
                "SELECT datasets.*, (SELECT count(*) FROM images WHERE dataset_id=datasets.id) AS image_count, "
                "(SELECT count(*) FROM vision_tasks WHERE dataset_id=datasets.id) AS task_count, "
                "(SELECT id FROM images WHERE dataset_id=datasets.id ORDER BY created_at,id LIMIT 1) AS cover_image_id "
                "FROM datasets WHERE project_id=? ORDER BY updated_at DESC,id DESC", (project_id,)
            )]

    def get_dataset(self, project_id, dataset_id, db=None):
        if db is None:
            with self.connect() as connection:
                return self.get_dataset(project_id, dataset_id, connection)
        self.get(project_id, db)
        row = db.execute(
            "SELECT datasets.*, (SELECT count(*) FROM images WHERE dataset_id=datasets.id) AS image_count, "
            "(SELECT count(*) FROM vision_tasks WHERE dataset_id=datasets.id) AS task_count "
            "FROM datasets WHERE id=? AND project_id=?", (dataset_id, project_id)
        ).fetchone()
        if row is None:
            raise ProjectError(404, "DATASET_NOT_FOUND", "数据集不存在、已被删除，或不属于当前项目。")
        return self.public(row)

    def create_dataset(self, project_id, values):
        name, description, name_key = validate_project(values, "数据集")
        dataset_id, now = uuid.uuid4().hex, timestamp()
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                self.get(project_id, db)
                db.execute(
                    "INSERT INTO datasets(id,project_id,name,name_key,description,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?)", (dataset_id, project_id, name, name_key, description, now, now)
                )
                return self.get_dataset(project_id, dataset_id, db)
        except sqlite3.IntegrityError as exc:
            if "datasets.project_id, datasets.name_key" not in str(exc):
                raise
            raise ProjectError(409, "DATASET_NAME_EXISTS", "当前项目中已有同名数据集，请换一个名称。", {
                "name": "数据集名称在同一项目内不能重复。"
            }) from exc

    def update_dataset(self, project_id, dataset_id, values):
        name, description, name_key = validate_project(values, "数据集")
        revision = validate_revision(values.get("revision"), "数据集")
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                dataset = self.get_dataset(project_id, dataset_id, db)
                if dataset["revision"] != revision:
                    raise ProjectError(409, "REVISION_CONFLICT", "数据集已在其他页面更新。当前输入已保留，请关闭编辑并刷新后重试。")
                db.execute(
                    "UPDATE datasets SET name=?,name_key=?,description=?,revision=revision+1,updated_at=? "
                    "WHERE id=? AND project_id=? AND revision=?",
                    (name, name_key, description, timestamp(), dataset_id, project_id, revision)
                )
                return self.get_dataset(project_id, dataset_id, db)
        except sqlite3.IntegrityError as exc:
            if "datasets.project_id, datasets.name_key" not in str(exc):
                raise
            raise ProjectError(409, "DATASET_NAME_EXISTS", "当前项目中已有同名数据集，请换一个名称。", {
                "name": "数据集名称在同一项目内不能重复。"
            }) from exc

    def delete_dataset(self, project_id, dataset_id, values):
        revision = validate_revision(values.get("revision"), "数据集")
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                dataset = self.get_dataset(project_id, dataset_id, db)
                if dataset["revision"] != revision:
                    raise ProjectError(409, "REVISION_CONFLICT", "数据集信息已变化，请关闭弹窗并刷新后重新确认。")
                if values.get("confirmation_name") != dataset["name"]:
                    raise ProjectError(422, "CONFIRMATION_REQUIRED", "请输入完整数据集名称以确认删除。", {
                        "confirmation_name": "输入内容与数据集名称不一致。"
                    })
                if dataset['task_count']:
                    raise ProjectError(409,'DATASET_IN_USE',f"数据集已关联 {dataset['task_count']} 个任务，请先处理关联任务。")
                if dataset['image_count']:
                    raise ProjectError(409, 'DATASET_NOT_EMPTY', f"数据集包含 {dataset['image_count']} 张图片，不能直接删除；请保留原图与关联记录。")
                db.execute("DELETE FROM datasets WHERE id=? AND project_id=?", (dataset_id, project_id))
                return dataset
        except sqlite3.IntegrityError as exc:
            raise ProjectError(409, "DATASET_IN_USE", "数据集被其他数据引用，请先处理关联内容。") from exc

