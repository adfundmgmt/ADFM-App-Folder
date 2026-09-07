"""Single-process durable job queue with restart recovery and request coalescing."""
import hashlib,json,logging,sqlite3,time,uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from adfm_engine.services import DataUnavailable
class JobQueue:
    def __init__(self,path,handlers):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handlers=handlers;self.lock=Lock();self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='adfm-job')
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,fingerprint TEXT,kind TEXT,arguments TEXT,status TEXT,result TEXT,error TEXT,created REAL,updated REAL)')
            db.execute('CREATE INDEX IF NOT EXISTS job_fingerprint ON jobs(fingerprint)')
            pending=db.execute("SELECT id FROM jobs WHERE status IN ('queued','running') ORDER BY created").fetchall()
            db.execute("UPDATE jobs SET status='queued' WHERE status='running'")
        for row in pending:self.pool.submit(self.run,row['id'])
    @contextmanager
    def db(self):
        db=sqlite3.connect(self.path,timeout=30);db.row_factory=sqlite3.Row
        try:yield db;db.commit()
        except Exception:db.rollback();raise
        finally:db.close()
    def submit(self,kind,arguments):
        if kind not in self.handlers:raise ValueError('Unknown job type.')
        payload=json.dumps(arguments,sort_keys=True);key=hashlib.sha256((kind+payload).encode()).hexdigest();now=time.time()
        with self.lock,self.db() as db:
            db.execute("DELETE FROM jobs WHERE status IN ('completed','failed') AND updated<?",(now-86400,))
            previous=db.execute("SELECT id FROM jobs WHERE fingerprint=? AND (status IN ('queued','running') OR (status='completed' AND updated>?)) ORDER BY created DESC LIMIT 1",(key,now-21600)).fetchone()
            if previous:job_id=previous['id']
            else:
                if db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]>=8:raise DataUnavailable('The analysis queue is busy. Please retry shortly.')
                job_id=uuid.uuid4().hex;db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',(job_id,key,kind,payload,'queued',None,None,now,now))
        if not previous:self.pool.submit(self.run,job_id)
        return self.get(job_id)
    def get(self,job_id):
        with self.db() as db:row=db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if row is None:raise DataUnavailable('This job has expired. Run the analysis again.')
        return {'id':row['id'],'status':row['status'],'result':json.loads(row['result']) if row['result'] else None,'error':row['error']}
    def run(self,job_id):
        with self.db() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
            if row is None:return
            db.execute("UPDATE jobs SET status='running',updated=? WHERE id=?",(time.time(),job_id))
        try:
            payload=json.dumps(self.handlers[row['kind']](**json.loads(row['arguments'])),allow_nan=False)
            with self.db() as db:db.execute("UPDATE jobs SET status='completed',result=?,updated=? WHERE id=?",(payload,time.time(),job_id))
        except Exception as exc:
            logging.exception('Analysis job failed')
            message=str(exc) if isinstance(exc,(DataUnavailable,ValueError)) else 'The analysis could not complete. Please retry.'
            with self.db() as db:db.execute("UPDATE jobs SET status='failed',error=?,updated=? WHERE id=?",(message,time.time(),job_id))
    def close(self):self.pool.shutdown(wait=False,cancel_futures=True)
