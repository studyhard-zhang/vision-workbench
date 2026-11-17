"""Dataset ownership, references, conflicts and migration behavior."""
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.server import create_server
from app.storage import ProjectError, Storage


class DatasetBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'workspace.db'
        self.store = Storage(self.path)
        self.first = self.store.create({'name': '项目 A'})
        self.second = self.store.create({'name': '项目 B'})

    def tearDown(self):
        self.temp.cleanup()

    def test_each_project_has_its_own_datasets_and_name_scope(self):
        first = self.store.create_dataset(self.first['id'], {'name': '相机 A', 'description': '第一批原图'})
        second = self.store.create_dataset(self.first['id'], {'name': '相机 B'})
        other = self.store.create_dataset(self.second['id'], {'name': '相机 A'})
        self.assertEqual({d['id'] for d in self.store.list_datasets(self.first['id'])}, {first['id'], second['id']})
        self.assertEqual(self.store.list_datasets(self.second['id']), [other])
        self.assertNotIn('task_type', first)

    def test_duplicate_normalized_name_and_failed_rename_leave_records_unchanged(self):
        first = self.store.create_dataset(self.first['id'], {'name': 'Camera A', 'description': '保留'})
        second = self.store.create_dataset(self.first['id'], {'name': 'Camera B'})
        with self.assertRaises(ProjectError) as result:
            self.store.create_dataset(self.first['id'], {'name': ' ＣＡＭＥＲＡ A '})
        self.assertEqual(result.exception.code, 'DATASET_NAME_EXISTS')
        with self.assertRaises(ProjectError):
            self.store.update_dataset(self.first['id'], second['id'], {'name': first['name'], 'revision': 1})
        self.assertEqual(self.store.get_dataset(self.first['id'], second['id']), second)
        self.assertEqual(len(self.store.list_datasets(self.first['id'])), 2)

    def test_stale_edits_and_stale_deletes_cannot_overwrite_new_data(self):
        dataset = self.store.create_dataset(self.first['id'], {'name': '批次一'})
        updated = self.store.update_dataset(self.first['id'], dataset['id'], {
            'name': '批次二', 'description': '多行\n说明', 'revision': 1,
        })
        for operation in [
            lambda: self.store.update_dataset(self.first['id'], dataset['id'], {'name': '旧页面', 'revision': 1}),
            lambda: self.store.delete_dataset(self.first['id'], dataset['id'], {'revision': 1, 'confirmation_name': dataset['name']}),
        ]:
            with self.assertRaises(ProjectError) as result:
                operation()
            self.assertEqual(result.exception.code, 'REVISION_CONFLICT')
        self.assertEqual(self.store.get_dataset(self.first['id'], dataset['id']), updated)

    def test_cross_project_ids_cannot_read_update_or_delete_a_dataset(self):
        dataset = self.store.create_dataset(self.first['id'], {'name': '不能串项目'})
        for operation in [
            lambda: self.store.get_dataset(self.second['id'], dataset['id']),
            lambda: self.store.update_dataset(self.second['id'], dataset['id'], {'name': '错误位置', 'revision': 1}),
            lambda: self.store.delete_dataset(self.second['id'], dataset['id'], {'revision': 1, 'confirmation_name': dataset['name']}),
        ]:
            with self.assertRaises(ProjectError) as result:
                operation()
            self.assertEqual(result.exception.code, 'DATASET_NOT_FOUND')
        self.assertEqual(self.store.get_dataset(self.first['id'], dataset['id']), dataset)

    def test_nonempty_project_is_protected_and_dataset_delete_requires_confirmation(self):
        dataset = self.store.create_dataset(self.first['id'], {'name': '保留数据集'})
        self.assertEqual(self.store.get(self.first['id'])['dataset_count'], 1)
        with self.assertRaises(ProjectError) as result:
            self.store.delete(self.first['id'], {'revision': 1, 'confirmation_name': self.first['name']})
        self.assertEqual(result.exception.code, 'PROJECT_NOT_EMPTY')
        with self.assertRaises(ProjectError):
            self.store.delete_dataset(self.first['id'], dataset['id'], {'revision': 1, 'confirmation_name': '错误名称'})
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.connect() as db:
                db.execute('DELETE FROM projects WHERE id=?', (self.first['id'],))
        self.store.delete_dataset(self.first['id'], dataset['id'], {'revision': 1, 'confirmation_name': dataset['name']})
        self.assertEqual(self.store.get(self.first['id'])['dataset_count'], 0)
        self.store.delete(self.first['id'], {'revision': 1, 'confirmation_name': self.first['name']})

    def test_validation_and_persistence_preserve_parent_and_literal_text(self):
        for name in ['', ' ', 'x' * 61, '批次\n换行', 42]:
            with self.assertRaises(ProjectError):
                self.store.create_dataset(self.first['id'], {'name': name})
        text = '<img src=x onerror=alert(1)>\n原样保留'
        dataset = self.store.create_dataset(self.first['id'], {'name': '原图', 'description': text})
        restored = Storage(self.path)
        self.assertEqual(restored.get_dataset(self.first['id'], dataset['id']), dataset)
        self.assertEqual(restored.get(self.first['id'])['dataset_count'], 1)
        self.assertEqual(restored.get(self.second['id'])['dataset_count'], 0)


class DatasetHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = create_server(0, Path(self.temp.name) / 'http.db')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.project = self.server.storage.create({'name': '接口项目'})
        self.other = self.server.storage.create({'name': '另一接口项目'})
        self.collection = '/api/projects/' + self.project['id'] + '/datasets'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def call(self, path, method='GET', body=None):
        request = Request(self.base + path, method=method,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={'Content-Type': 'application/json'})
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def test_real_http_dataset_crud_and_project_protection(self):
        status, data = self.call(self.collection, 'POST', {'name': '批次 A'})
        self.assertEqual(status, 201)
        dataset = data['dataset']
        path = self.collection + '/' + dataset['id']
        self.assertEqual(self.call(self.collection)[1]['total'], 1)
        self.assertEqual(self.call(path)[1]['dataset'], dataset)
        status, changed = self.call(path, 'PUT', {'name': '批次 B', 'description': '新说明', 'revision': 1})
        self.assertEqual(status, 200)
        self.assertEqual(changed['dataset']['revision'], 2)
        self.assertEqual(self.call('/api/projects/' + self.project['id'], 'DELETE', {
            'revision': 1, 'confirmation_name': self.project['name'],
        })[0], 409)
        self.assertEqual(self.call(path, 'DELETE', {'revision': 2, 'confirmation_name': '批次 B'})[0], 200)
        self.assertEqual(self.call(path)[0], 404)
        self.assertEqual(self.call('/api/projects/' + self.project['id'])[1]['project']['dataset_count'], 0)

    def test_cross_project_and_missing_project_requests_fail_without_writes(self):
        dataset = self.server.storage.create_dataset(self.project['id'], {'name': '本项目素材'})
        wrong = '/api/projects/' + self.other['id'] + '/datasets/' + dataset['id']
        for method, body in [('GET', None), ('PUT', {'name': '修改', 'revision': 1}),
                             ('DELETE', {'confirmation_name': dataset['name'], 'revision': 1})]:
            self.assertEqual(self.call(wrong, method, body)[0], 404)
        self.assertEqual(self.call('/api/projects/' + '0' * 32 + '/datasets', 'POST', {'name': '无项目'})[0], 404)
        self.assertEqual(self.server.storage.get_dataset(self.project['id'], dataset['id']), dataset)

    def test_type_and_parent_override_fields_are_rejected(self):
        for extra in [{'task_type': 'detect'}, {'project_id': self.other['id']}]:
            self.assertEqual(self.call(self.collection, 'POST', {'name': '不应创建', **extra})[0], 422)
        self.assertEqual(self.call(self.collection)[1]['total'], 0)


class DatasetMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'v2.db'
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('CREATE TABLE schema_version(version INTEGER PRIMARY KEY)')
            db.execute('INSERT INTO schema_version VALUES(2)')
            Storage.create_project_table(db, 'projects')
            db.execute('INSERT INTO projects VALUES(?,?,?,?,?,?,?)', ('b' * 32, '用户项目', '用户项目', '说明', 4, '2026-10-27', '2026-10-27'))
            db.commit()

    def tearDown(self):
        self.temp.cleanup()

    def test_v2_upgrade_preserves_projects_backups_and_allows_dataset_creation(self):
        store = Storage(self.path)
        project = store.get('b' * 32)
        self.assertEqual((project['name'], project['description'], project['revision']), ('用户项目', '说明', 4))
        self.assertEqual(project['dataset_count'], 0)
        store.create_dataset(project['id'], {'name': '新素材'})
        self.assertEqual(Storage(self.path).get(project['id'])['dataset_count'], 1)
        backups = list((self.path.parent / 'backups').glob('schema-v2-*.db'))
        self.assertEqual(len(backups), 1)
        with closing(sqlite3.connect(backups[0])) as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT count(*) FROM projects').fetchone()[0], 1)

    def test_failed_dataset_migration_rolls_back_to_v2(self):
        class FailingStorage(Storage):
            @staticmethod
            def create_dataset_table(db):
                Storage.create_dataset_table(db)
                raise RuntimeError('Injected dataset migration failure')
        with self.assertRaises(RuntimeError):
            FailingStorage(self.path)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT count(*) FROM projects').fetchone()[0], 1)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='datasets'").fetchone())


if __name__ == '__main__':
    unittest.main()
