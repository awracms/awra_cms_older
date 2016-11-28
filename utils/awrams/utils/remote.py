import paramiko
import os
from getpass import getpass
import re
import copy

class Interactor:
    def __init__(self,ssh_client):
        self.client = ssh_client
        
    def __call__(self,command):
        stdin, stdout, stderr = self.client.exec_command(command)
        err = [k.strip() for k in stderr.readlines()]
        return [k.strip() for k in stdout.readlines()], err
        
    def put(self,localfile,remotepath=None):
        if remotepath is None:
            remotepath = os.path.split(localfile)[-1]
        sftp = self.client.open_sftp()
        sftp.put(localfile, remotepath)
        sftp.close()
        
    def get(self,remotefile,localpath=None):
        if localpath is None:
            localpath = os.path.split(remotefile)[-1]
        sftp = self.client.open_sftp()
        sftp.get(remotefile,localpath)
        sftp.close()

def establish_remote_session(username=None,host='raijin.nci.org.au'):
    client = paramiko.SSHClient()

    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if username is None:
        username=input("Username:")

    client.connect(host, username=username,password=getpass('Password:'))
    return Interactor(client)

def clone_to_raijin(session,remote_path,force=False,local_path=None,exclusions=None):
    '''
    No error checking! This is for convenience only...
    '''
    from subprocess import call
    from os import environ, chdir

    if local_path is None:
        local_path=environ['AWRAPATH']

    os.chdir(local_path)

    if exclusions is None:
        exclusions = ['.git','AcceptanceTests','Samples','*ipynb*','WIRADA','River','*__pycache__*']

    tar_cmd = 'tar -cf out.tgz *'
    for k in exclusions:
        tar_cmd = tar_cmd + ' --exclude='+k

    call(tar_cmd,shell=True)

    if force:
        session('rm -rf %s' % remote_path)

    session('mkdir -p %s' % remote_path)
    session.put('out.tgz',remote_path + '/out.tgz')

    session('cd %s; tar -xf out.tgz; rm out.tgz' % remote_path)
    call('rm out.tgz',shell=True)

    session('cd %s; mv raijin_activate.sh activate' % remote_path)
    session('cd %s/Config/; mv raijin_host_defaults.py host_defaults.py' % remote_path)


class HostPath:
    def __init__(self,key,relpath=''):
        self.key = key
        self.relpath = relpath
        
    def __call__(self,relpath=''):
        return HostPath(self,relpath)
        
    def localise(self,pathmap):
        if isinstance(self.key,HostPath):
            mapped = self.key.localise(pathmap)
        elif isinstance(pathmap[self.key],HostPath):
            mapped = pathmap[self.key].localise(pathmap)
        else:
            mapped = pathmap[self.key]
        return os.path.join(mapped,self.relpath)

    def __repr__(self):
        return repr(self.key)+'/'+self.relpath

def resolve_hostpaths(node,host_dict=None,parent=None,key=None,copydict=True):
    '''
    Localise a full map of hostpaths
    '''
    if host_dict is None:
        host_dict = node
    if parent is None and key is None and copydict == True:
        node = copy.deepcopy(node)
    if hasattr(node,'items'):
        for k,v in node.items():
            resolve_hostpaths(v,host_dict,node,k)
    else:
        if isinstance(node,HostPath):
            parent[key] = node.localise(host_dict)
    return node

class PBSJob:
    def __init__(self,client,job_name,pbs_name,outpath):
        self.client = client
        self.job_name = job_name
        self.pbs_name = pbs_name
        self.pbs_id = pbs_name.split('.')[0]
        self.outpath = outpath
        
    def get_status(self):
        status_str, error_str = self.client('qstat %s' % self.pbs_name)
        if len(error_str) > 0:
            if re.match('.*Job has finished.*',str(error_str)) is not None:
                return 'Finished'
        else:
            return status_str
                
    def get_output(self):
        res,err = self.client('qcat %s' % self.pbs_name)
        if len(res) > 0:
            if res[0] == 'PBS error: Job has finished':
                res = self.client('cat %s' % os.path.join(self.outpath,self.job_name+'.o'+self.pbs_id))
        return res
        
    def get_errors(self):
        res,err = self.client('qcat -e %s' % self.pbs_name)
        if len(res) > 0:
            if res[0] == 'PBS error: Job has finished':
                res = self.client('cat %s' % os.path.join(self.outpath,self.job_name+'.e'+self.pbs_id))
        return res

    def cancel(self):
        return self.client('qdel %s' % self.pbs_name)
