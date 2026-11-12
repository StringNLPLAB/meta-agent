"""Bounded directory-FD observations of editable training files."""
import hashlib
import os
from pathlib import Path
import stat
import time

def _sha(raw): return hashlib.sha256(raw).hexdigest()

def _directory(path):
    p=Path(path).absolute()
    if p.resolve(strict=True)!=p or not p.is_dir():
        raise ValueError("Actual live paths must be canonical existing directories")
    return p

def live_census(roots,*,maximum_files=10000,maximum_file_bytes=67108864,maximum_total_bytes=268435456):
    """Observe actual names through directory FDs; never follow live symlinks."""
    result=dict(started_at=time.time(),roots=list(map(str,roots)),files={},errors=[],complete=True)
    total=0
    for root in roots:
        pending=[]
        try:
            # All descendant opens are relative to pinned directory descriptors.
            # Renaming a directory to a symlink cannot redirect the host reader.
            descriptor=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            pending=[(Path(root),descriptor)]
            while pending:
                directory,parent_fd=pending.pop()
                try:
                    with os.scandir(parent_fd) as stream:
                        names=sorted(entry.name for entry in stream)
                    for name in names:
                        path=directory/name;key=str(path)
                        info=os.stat(name,dir_fd=parent_fd,follow_symlinks=False);mode=info.st_mode
                        if stat.S_ISDIR(mode):
                            child_fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd)
                            pending.append((path,child_fd));continue
                        if len(result['files'])>=maximum_files:raise ValueError("Census file limit")
                        row=dict(observed_at=time.time(),bytes=info.st_size,mtime_ns=info.st_mtime_ns)
                        if stat.S_ISLNK(mode):
                            target=os.readlink(name,dir_fd=parent_fd)
                            row.update(kind="symlink",target=target,sha256=_sha(target.encode()))
                        elif stat.S_ISREG(mode):
                            row['kind']='file'
                            if info.st_size>maximum_file_bytes or total+info.st_size>maximum_total_bytes:
                                raise ValueError("Census byte limit")
                            file_fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent_fd)
                            with os.fdopen(file_fd,'rb') as stream:
                                opened=os.fstat(stream.fileno())
                                if not stat.S_ISREG(opened.st_mode):raise ValueError("Live file changed type")
                                digest=hashlib.sha256();read=0
                                while chunk:=stream.read(1048576):
                                    read+=len(chunk)
                                    if read>maximum_file_bytes or total+read>maximum_total_bytes:raise ValueError("Census byte limit")
                                    digest.update(chunk)
                            total+=read;row.update(sha256=digest.hexdigest(),bytes=read)
                            after=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
                            if (after.st_ino,after.st_size,after.st_mtime_ns)!=(info.st_ino,info.st_size,info.st_mtime_ns):
                                result['errors'].append(dict(path=key,error='FileChangedDuringObservation'));result['complete']=False
                        else:row.update(kind='special',sha256=None)
                        result['files'][key]=row
                    if directory.resolve()!=directory:
                        result['errors'].append(dict(path=str(directory),error='DirectoryPathChangedDuringObservation'));result['complete']=False
                finally:
                    os.close(parent_fd)
        except Exception as error:
            result['errors'].append(dict(path=str(root),error=type(error).__name__))
            result['complete']=False
        finally:
            for _,descriptor in pending:os.close(descriptor)
    result.update(finished_at=time.time(),total_file_bytes_hashed=total)
    return result
