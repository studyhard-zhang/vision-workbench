def drop_inferences(db,restore_legacy_guards=True):
    for kind in ('training','evaluation','inference'):db.execute(f'DROP TRIGGER IF EXISTS {kind}_shared_compute_guard')
    for table in ('inference_job_inputs','inference_remote_sessions','inference_dispatches','inference_jobs','inference_upload_failures','inference_inputs'):db.execute('DROP TABLE IF EXISTS '+table)
    if restore_legacy_guards:
        for first,other in (('training','evaluation'),('evaluation','training')):
            db.execute(f"CREATE TRIGGER {first}_shared_compute_guard BEFORE INSERT ON {first}_dispatches WHEN NEW.state='AWAITING_RESULT' AND EXISTS(SELECT 1 FROM {other}_dispatches WHERE target_id=NEW.target_id AND state='AWAITING_RESULT') BEGIN SELECT RAISE(ABORT,'Compute target is reserved by another task kind'); END")


def drop_evaluations(db):
    """Reconstruct an authentic pre-evaluation schema in legacy migration fixtures."""
    drop_inferences(db,False)
    db.execute('DROP TABLE evaluation_remote_sessions')
    db.execute('DROP TABLE evaluation_dispatches')
    db.execute('DROP TABLE evaluation_jobs')
    db.execute('DROP INDEX model_task_identity')
