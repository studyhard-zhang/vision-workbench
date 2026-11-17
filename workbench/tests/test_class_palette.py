from test_annotations import AnnotationFixture
from app.annotations import CLASS_COLORS
from app.storage import ProjectError


class ClassPaletteTests(AnnotationFixture):
    def test_palette_is_task_scoped_and_rename_does_not_recolor(self):
        ids=[self.add(name=f'类别 {i}') for i in range(3)]
        self.add(self.b,name='独立类别')
        self.assertEqual([row['color'] for row in self.repo.classes(self.pid,self.a)],list(CLASS_COLORS[:3]))
        self.assertEqual(self.repo.classes(self.pid,self.b)[0]['color'],CLASS_COLORS[0])
        self.repo.change_class(self.pid,self.a,{'name':'新名称','revision':1},ids[1])
        self.assertEqual(self.repo.classes(self.pid,self.a)[1]['color'],CLASS_COLORS[1])

    def test_deletion_does_not_recolor_remaining_classes_and_free_color_can_be_reused(self):
        first=self.add(name='A');self.add(name='B')
        self.repo.change_class(self.pid,self.a,{'revision':1,'confirmation_name':'A'},first,True)
        self.add(name='C')
        self.assertEqual([row['color'] for row in self.repo.classes(self.pid,self.a)],[CLASS_COLORS[1],CLASS_COLORS[0]])

    def test_custom_color_is_rejected_and_existing_annotations_survive_palette_adoption(self):
        cid=self.add();self.save('annotated',boxes=[self.box(cid)])
        with self.assertRaises(ProjectError):self.repo.change_class(self.pid,self.a,{'name':'颜色','color':'#123456'})
        before=self.repo.get(self.pid,self.a,self.image)
        with self.store.connect() as db:db.execute('UPDATE task_classes SET color=? WHERE id=?',('#000000',cid))
        self.assertEqual(self.repo.adopt_default_colors(),1)
        self.assertEqual(self.repo.get(self.pid,self.a,self.image),before)
        self.assertEqual(self.repo.classes(self.pid,self.a)[0]['color'],CLASS_COLORS[0])
        self.assertEqual(self.repo.adopt_default_colors(),0)
