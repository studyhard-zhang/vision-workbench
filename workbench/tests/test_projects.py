"""Behavior checks for persistence, revision conflicts and HTTP boundaries."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.server import create_server
from app.storage import ProjectError, Storage


class ProjectBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "workspace.db"
        self.store = Storage(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_project_survives_new_storage_connection_and_preserves_literal_content(self):
        text = '<script>alert("project")</script>\n第一批数据'
        project = self.store.create({"name": "零件检测", "description": text})
        restored = Storage(self.path).get(project["id"])
        self.assertEqual(restored["description"], text)
        self.assertNotIn("task_type", restored)
        self.assertEqual(restored["revision"], 1)

    def test_duplicate_normalized_name_is_rejected_without_creating_extra_project(self):
        self.store.create({"name": "Parts A"})
        with self.assertRaises(ProjectError) as result:
            self.store.create({"name": " ＰＡＲＴＳ A "})
        self.assertEqual(result.exception.code, "NAME_EXISTS")
        self.assertEqual(len(self.store.list_projects()), 1)

    def test_two_editors_cannot_silently_overwrite_each_other(self):
        original = self.store.create({"name": "原项目"})
        self.store.update(original["id"], {"name": "已更新", "revision": 1})
        with self.assertRaises(ProjectError) as result:
            self.store.update(original["id"], {"name": "旧页面修改", "revision": 1})
        self.assertEqual(result.exception.code, "REVISION_CONFLICT")
        self.assertEqual(self.store.get(original["id"])["name"], "已更新")

    def test_deletion_requires_current_revision_and_exact_confirmation(self):
        project = self.store.create({"name": "待删除项目"})
        with self.assertRaises(ProjectError):
            self.store.delete(project["id"], {"revision": 1, "confirmation_name": "错误名称"})
        self.store.update(project["id"], {"name": "重命名项目", "revision": 1})
        with self.assertRaises(ProjectError) as result:
            self.store.delete(project["id"], {"revision": 1, "confirmation_name": "待删除项目"})
        self.assertEqual(result.exception.code, "REVISION_CONFLICT")
        self.store.delete(project["id"], {"revision": 2, "confirmation_name": "重命名项目"})
        self.assertEqual(self.store.list_projects(), [])

    def test_invalid_content_and_failed_rename_preserve_saved_records(self):
        first = self.store.create({"name": "项目一", "description": "需要保留"})
        self.store.create({"name": "项目二"})
        with self.assertRaises(ProjectError):
            self.store.update(first["id"], {"name": "项目二", "revision": 1})
        self.assertEqual(self.store.get(first["id"]), first)
        for name in [" ", "x" * 61, "项目\n换行", 42]:
            with self.assertRaises(ProjectError):
                self.store.create({"name": name})
        self.assertEqual(len(self.store.list_projects()), 2)


class SchemaMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'legacy.db'
        self.original = ('a' * 32, '已有项目', '已有项目', '用户原说明', 'detect', 7,
                         '2026-10-27T01:00:00+00:00', '2026-10-27T02:00:00+00:00')
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript("""
                CREATE TABLE schema_version(version INTEGER PRIMARY KEY);
                INSERT INTO schema_version VALUES(1);
                CREATE TABLE projects(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL DEFAULT 'detect' CHECK(task_type='detect'),
                    revision INTEGER NOT NULL DEFAULT 1 CHECK(revision>0),
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
            """)
            db.execute('INSERT INTO projects VALUES(?,?,?,?,?,?,?,?)', self.original)
            db.commit()

    def tearDown(self):
        self.temp.cleanup()

    def test_migration_removes_project_task_type_and_keeps_all_user_fields_and_backup(self):
        store = Storage(self.path)
        actual = store.get('a' * 32)
        self.assertEqual(actual, dict(zip(
            ['id', 'name', 'description', 'revision', 'created_at', 'updated_at'],
            [self.original[i] for i in [0, 1, 3, 5, 6, 7]],
        )) | {'dataset_count': 0, 'task_count': 0})
        with store.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0], 16)
            self.assertNotIn('task_type', [r[1] for r in db.execute('PRAGMA table_info(projects)')])
        backups = list((self.path.parent / 'backups').glob('schema-v1-*.db'))
        self.assertEqual(len(backups), 1)
        with closing(sqlite3.connect(backups[0])) as db:
            self.assertEqual(db.execute('SELECT * FROM projects').fetchone(), self.original)

    def test_reopening_migrated_database_is_idempotent_and_existing_project_can_be_edited(self):
        Storage(self.path)
        store = Storage(self.path)
        self.assertEqual(len(list((self.path.parent / 'backups').glob('schema-v1-*.db'))), 1)
        updated = store.update('a' * 32, {'name': '业务项目', 'description': '多数据集', 'revision': 7})
        self.assertEqual(updated['revision'], 8)
        self.assertNotIn('task_type', store.create({'name': '新业务项目'}))

    def test_failed_migration_rolls_back_without_losing_legacy_project(self):
        class FailingStorage(Storage):
            @staticmethod
            def create_project_table(db, table_name):
                Storage.create_project_table(db, table_name)
                if table_name == 'projects_v2':
                    raise RuntimeError('Injected migration failure')
        with self.assertRaises(RuntimeError):
            FailingStorage(self.path)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT * FROM projects').fetchone(), self.original)
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0], 1)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='projects_v2'").fetchone())


class HttpBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.server = create_server(0, Path(cls.temp.name) / "test.db")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        cls.temp.cleanup()

    def call(self, path, body=None, method=None, headers=None):
        request = Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                          method=method, headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def test_real_http_crud_roundtrip(self):
        status, data = self.call('/api/projects', {"name": "接口试用", "description": "持久化"})
        self.assertEqual(status, 201)
        project = data['project']
        status, updated = self.call('/api/projects/' + project['id'],
                                    {"name": "接口试用更新", "description": "修改已保存", "revision": 1}, 'PUT')
        self.assertEqual(status, 200)
        self.assertEqual(updated['project']['revision'], 2)
        status, saved = self.call('/api/projects/' + project['id'])
        self.assertEqual(saved['project']['description'], "修改已保存")
        status, _ = self.call('/api/projects/' + project['id'],
                              {"revision": 2, "confirmation_name": "接口试用更新"}, 'DELETE')
        self.assertEqual(status, 200)
        self.assertEqual(self.call('/api/projects/' + project['id'])[0], 404)

    def test_cross_origin_mutation_and_hidden_paths_are_rejected(self):
        status, data = self.call('/api/projects', {"name": "外部请求"}, headers={"Origin": "https://other.example"})
        self.assertEqual((status, data['error']['code']), (403, 'ORIGIN_NOT_ALLOWED'))
        for path in ['/var/platform.db', '/app/storage.py', '/../../run.py']:
            self.assertEqual(self.call(path)[0], 404)

    def test_rejected_body_returns_error_without_creating_records(self):
        before=self.call('/api/projects')[1]
        for _ in range(5):
            status,data=self.call('/api/projects',{'name':'被拒绝的数据','description':'x'*100000},headers={'Origin':'https://other.example'})
            self.assertEqual((status,data['error']['code']),(403,'ORIGIN_NOT_ALLOWED'))
        self.assertEqual(self.call('/api/projects')[1],before)
        self.assertEqual(self.call('/api/health')[0],200)

    def test_unsupported_fields_and_content_type_do_not_create_projects(self):
        self.assertEqual(self.call('/api/projects', {"name": "异常字段", "task_type": "segment"})[0], 422)
        self.assertEqual(self.call('/api/projects', {"name": "文本提交"}, headers={"Content-Type": "text/plain"})[0], 415)


class ProcessPersistenceTests(unittest.TestCase):
    def test_saved_project_survives_server_process_restart(self):
        app_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as data_dir:
            def start():
                process = subprocess.Popen(
                    [sys.executable, str(app_root / 'run.py'), '--port', '0', '--data-dir', data_dir],
                    cwd=app_root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                )
                line = process.stdout.readline().strip()
                if not line.startswith('Local: http://127.0.0.1:'):
                    process.terminate()
                    process.wait(timeout=5)
                    process.stdout.close()
                    raise AssertionError('Test server failed to start')
                return process, line.removeprefix('Local: ').rstrip('/')

            def stop(process):
                process.terminate()
                process.wait(timeout=5)
                process.stdout.close()

            first, base = start()
            try:
                request = Request(base + '/api/projects',
                                  data=json.dumps({'name': '重启保留验证', 'description': '原说明'}).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=3) as response:
                    created = json.loads(response.read())['project']
                request = Request(base + '/api/projects/' + created['id'] + '/datasets',
                                  data=json.dumps({'name': '原图批次 A', 'description': '数据集也应保留'}).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=3) as response:
                    dataset = json.loads(response.read())['dataset']
            finally:
                stop(first)
            second, base = start()
            try:
                with urlopen(base + '/api/projects/' + created['id'], timeout=3) as response:
                    restored = json.loads(response.read())['project']
                self.assertEqual(restored, {**created, 'dataset_count': 1})
                with urlopen(base + '/api/projects/' + created['id'] + '/datasets/' + dataset['id'], timeout=3) as response:
                    self.assertEqual(json.loads(response.read())['dataset'], dataset)
            finally:
                stop(second)


if __name__ == '__main__':
    unittest.main()
