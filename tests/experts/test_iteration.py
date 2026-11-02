"""Line indexes, per-attempt search trees, the site and state-only media preserve source evidence."""
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import numpy as np
import pytest

from table_1000.experts.audit import iteration, site
from table_1000.experts.audit.codec import save_document
from table_1000.experts.audit.codec import load_document


def batch(path, *, outcome='success', category=None, source_step=1, names=None, repeat=0, parent=None, a1=True):
    """A batch with facts: a caught planning failure (a1) unless ``a1`` is False, and for
    ``category`` a check whose outcome makes it a2 (step passed), a3 (rolled back to 0 on
    main/0-1) or pending (no outcome)."""
    names = names or ['store pen']
    (path/'episodes').mkdir(parents=True, exist_ok=True)
    summary = dict(run=path.name, scene='000100', robot='x1pro', expert='0000', layout=0, commit=path.name[-1],
                   physics='isaac-cpu', steps_run=[source_step, source_step],
                   **{'from': f'../{parent.name}' if parent else None})
    iteration.write(path/'summary.json', summary)
    report = dict(seed=0, repeat=repeat, status=outcome, control_steps=4, retry_archives=[],
        steps=[dict(step=source_step, name='store pen', at=4, cpu_s=3., wall_s=4.)] if outcome=='success' else [],
        step_names=names, call_trace=f'calls/0{"-"+str(repeat) if repeat else ""}.json.gz',
        **{'from': 'parent/checkpoints/step0.hdf5'})
    iteration.write(path/'episodes'/f'0{"-"+str(repeat) if repeat else ""}.json', report)
    frames = [dict(id=0, parent=None, kind='step', name=names[source_step-1], arguments={'index': source_step},
                   enter=0, exit=4, branch='main', concrete=False, control=False, cpu_s=3., wall_s=4.),
              dict(id=1, parent=0, kind='skill', name='pick pen', arguments={'obj':'pen'}, enter=0, exit=0,
                   branch='main', concrete=True, control=False, cpu_s=.5, wall_s=.6,
                   **(dict(exception=dict(kind='planning', reason='no IK'), failure_id=1) if a1 else {})),
              dict(id=2, parent=0, kind='skill', name='pick pen', arguments={'obj':'pen'}, enter=0, exit=4,
                   branch='main', concrete=True, control=False, cpu_s=2., wall_s=3.)]
    failures = [dict(id=1, kind='planning', frame=1, branch='main', at=0, step=0, reason='no IK', stage='release IK')] if a1 else []
    outcomes = []
    if category:
        failures.append(dict(id=4, kind='check', frame=2, branch='main', at=2, start=0, step=0, reason='missed pen',
                             criterion_id='grasp.rise', measurements=[dict(name='rise', actual=.001, required=.01)]))
    if category == 'a3':
        outcomes.append(dict(id=0, kind='rollback', step=0, branch='main', at=2, control=2, failure_id=4, discarded=[4],
                             restore=dict(branch='main/2-1', at=0), cost_start=[0., 0.], reason='missed pen'))
        frames.append(dict(frames[2], id=3, branch='main/2-1'))
    if category != 'pending' and outcome == 'success':
        outcomes.append(dict(id=1, kind='step_passed', step=0, frame=0, branch=frames[-1]['branch'], at=4))
    elif outcome != 'success':
        outcomes.append(dict(id=1, kind='step_abandoned', step=0, frame=0, branch='main', at=4, start=0, reason='failed'))
    save_document(path/report['call_trace'], dict(version=2, frames=frames, failures=failures, checks=[], outcomes=outcomes,
                                                  solves=[], decisions=[], solver_coverage='none'))
    with h5py.File(path/'trajectories.hdf5','w') as f:
        data=f.create_group('data');data.attrs['env_args']=json.dumps(dict(scene='000100',robot='x1pro'))
        g=data.create_group('demo_0');g.attrs['num_samples']=4
        g.create_dataset('initial_state/x',data=[[10]])
        g.create_dataset('states/x',data=np.array([[11],[12],[13],[14]]))
    return path


@pytest.fixture
def line(tmp_path):
    """runs/expert/<scene>/<robot>/<NNNN>/ of a test."""
    return tmp_path/'runs'/'000100'/'x1pro'/'0000'


def joined(line, parts, name='20260102-000000-joined', steps=1):
    path = line/name
    batch(path)
    summary = iteration.read(path/'summary.json')
    summary['joined'] = [f'../{p.name}' for p in parts]
    summary['steps'] = dict(seeds=1, rows=[dict(step=k + 1) for k in range(steps)])
    iteration.write(path/'summary.json', summary)
    return path


def test_index_merges_outer_attempts_and_counts_a1_without_promoting_pending(line):
    first=batch(line/'20260101-120000-a',category='pending')
    last=batch(line/'20260101-121000-b',category='a2')
    joined(line,[last])
    step=iteration.line_index(line)['steps'][0]
    assert len(step['attempts'])==2 and len(step['nodes'])==6
    assert sum(n['a1_count'] for n in step['nodes'])==2  # one caught planning failure per batch
    assert all(b['count']==1 and b['reason']=='no IK' for n in step['nodes'] for b in n['a1'])
    assert {'pending','a2'} <= {e['category'] for e in step['edges']}
    assert [a['cpu_s'] for a in step['attempts']]==[3.,3.]
    assert [a['selected'] for a in step['attempts']]==[False,True]
    assert [a['batch'] for a in step['attempts']]==[first.name,last.name]
    assert (line/'.iteration/index.json').is_file()


def test_the_joined_batch_covering_most_steps_decides_the_chain_whatever_its_name(line):
    first=batch(line/'20260101-120000-a',names=['open','store pen'])
    second=batch(line/'20260101-130000-b',names=['open','store pen'],source_step=2,parent=first)
    joined(line,[first,second],name='complete-joined',steps=2)
    joined(line,[first],name='prefix1-joined')
    assert iteration.batches(line)[1].name=='complete-joined'


def test_without_a_joined_batch_the_chain_follows_from_parents(line):
    first=batch(line/'20260101-120000-a',names=['open','store pen'])
    second=batch(line/'20260101-130000-b',names=['open','store pen'],source_step=2,parent=first)
    batch(line/'20260101-140000-c',names=['open','store pen'],outcome='failed',category='pending')
    index=iteration.line_index(line)
    assert index['chain']==[first.name,second.name]
    assert [[a['selected'] for a in s['attempts']] for s in index['steps']]==[[True,False],[True]]


def test_name_mapping_survives_renumbering(line):
    batch(line/'20260101-120000-old',source_step=2,names=['old step','store pen'])
    batch(line/'20260101-130000-new',names=['store pen'])
    step=iteration.line_index(line)['steps'][0]
    assert step['name']=='store pen' and [a['source_step'] for a in step['attempts']]==[2,1]


def test_renamed_step_is_deprecated_beside_its_replacement(line):
    batch(line/'20260101-120000-old',names=['open','put tape in drawer','close'],source_step=2)
    batch(line/'20260101-130000-new',names=['open','put tape in basket','close'],source_step=2)
    result=iteration.line_index(line)
    assert [(s['step'],s['name'],s['deprecated']) for s in result['steps']]==[
        (1,'open',False),(None,'put tape in drawer',True),(2,'put tape in basket',False),(3,'close',False)]
    deprecated=result['steps'][1]
    assert deprecated['replacement']=={'candidates':['put tape in basket'],'certain':True,'previous':'open'}
    assert deprecated['source_steps']==[2] and len(deprecated['attempts'])==1
    assert iteration._replacements('b',[['a','b','c'],['a','x','y','c']])['certain'] is False
    assert iteration._keys(['a','b','a'])==['a#1','b','a#2']


def test_repeat_identities_are_retained(line):
    path=batch(line/'20260101-120000-a')
    batch(path,repeat=1)
    step=iteration.line_index(line)['steps'][0]
    assert sum(n['a1_count'] for n in step['nodes'])==2
    assert len({n['id'] for n in step['nodes']})==len(step['nodes'])
    assert [a['repeat'] for a in step['attempts']]==[0,1]


def test_zero_frame_step(line):
    source=batch(line/'20260101-120000-a')
    tree=load_document(source/'calls/0.json.gz')
    tree['frames']=tree['frames'][:1]
    tree['frames'][0].update(enter=0,exit=0)
    tree['failures']=[]
    save_document(source/'calls/0.json.gz',tree)
    assert len(iteration.line_index(line)['steps'][0]['nodes'])==1


def test_graphs_are_cached_until_a_batch_changes(line,monkeypatch):
    source=batch(line/'20260101-120000-a')
    iteration.line_index(line)
    assert not iteration.stale(line)
    calls=[]
    native=iteration._batch_graph
    monkeypatch.setattr(iteration,'_batch_graph',lambda *a:calls.append(a) or native(*a))
    iteration.line_index(line)
    assert calls==[]
    batch(line/'20260101-130000-b')
    assert iteration.stale(line)
    iteration.line_index(line)
    assert [c[0].name for c in calls]==['20260101-130000-b']
    report=iteration.read(source/'episodes/0.json');report['status']='failed'
    iteration.write(source/'episodes/0.json',report)
    assert iteration.stale(line)


def archived(line, **options):
    """A batch whose check at 2 rolled back to 0; its discarded main branch is archived."""
    source=batch(line/'20260101-120000-a',category='a3',**options)
    archive=batch(source/'attempts/0/0000-2-1',category='a3',**options)
    report=iteration.read(source/'episodes/0.json')
    report['retry_archives']=[dict(batch='attempts/0/0000-2-1',branch='main',control=2,retry=1,failure_id=4)]
    iteration.write(source/'episodes/0.json',report)
    # The discarded skill stopped at the check (2); the archive's trajectory ends there.
    tree=load_document(source/'calls/0.json.gz');tree['frames'][2]['exit']=2
    save_document(source/'calls/0.json.gz',tree)
    old=load_document(archive/'calls/0.json.gz')
    old['frames']=[dict(f,exit=min(f['exit'],2)) for f in old['frames'] if f['branch']=='main']
    old['outcomes']=[o for o in old['outcomes'] if o['kind']=='rollback']
    save_document(archive/'calls/0.json.gz',old)
    record=iteration.read(archive/'episodes/0.json');record['control_steps']=2
    iteration.write(archive/'episodes/0.json',record)
    return source, archive


def test_physical_rollback_ends_the_discarded_branch_and_forks_at_the_restored_state(line):
    archived(line)
    step=iteration.line_index(line)['steps'][0]
    rollback=next(e for e in step['edges'] if e['category']=='rollback')
    assert rollback['source'].endswith('/main@2') and rollback['target'].endswith('/main/2-1@0')
    assert rollback['trigger']=='a3' and rollback['failures'][0]['category']=='a3'
    assert any(e['category']=='a3' for e in step['edges'])
    [tree]=iteration.forest(step)
    parents={e['target']:e['source'] for e in tree['edges']}
    assert len(parents)==len(tree['edges'])
    assert next(n for n in tree['nodes'] if n['id'].endswith('/main@2'))['terminal']=='a3'
    assert all(not n['id'].endswith('main/2-1@0') for n in tree['nodes'])  # merged into the restored state
    # Counted per failure, not per colored edge.
    assert tree['failure_counts']=={'a1':1,'a3':1} and sorted(f['category'] for f in step['failures'])==['a1','a3']


def test_planning_rollback_is_marked_as_such(line):
    source,_=archived(line)
    tree=load_document(source/'calls/0.json.gz')
    tree['failures']=[dict(f,kind='planning') if f['id']==4 else f for f in tree['failures']]
    save_document(source/'calls/0.json.gz',tree)
    step=iteration.line_index(line)['steps'][0]
    assert next(e for e in step['edges'] if e['category']=='rollback')['trigger']=='planning'


def test_discarded_branch_uses_archive_even_when_control_numbers_fit_final_trajectory(line):
    source,archive=archived(line,a1=False)
    with h5py.File(archive/'trajectories.hdf5','r+') as f:
        f['data/demo_0/initial_state/x'][...]=[[100]]
        f['data/demo_0/states/x'][...]=[[101],[102],[103],[104]]
    index=iteration.line_index(line)
    main=[n for n in index['steps'][0]['nodes'] if n['branch'].endswith('/main')]
    assert main and all(n['batch']==f'{source.name}/attempts/0/0000-2-1' for n in main)
    span=next(s for s in index['media'] if s['kind']=='still' and s['start']==2 and 'attempts' in s['batch'])
    [prepared]=iteration.prepare_media(line,[span],line.parent/'media')
    with h5py.File(line.parent/'media/trajectories.hdf5') as f:
        assert f[f'data/demo_{prepared["seed"]}/initial_state/x'][()].tolist()==[[102]]


def test_enclosing_step_exit_after_restore_belongs_to_resumed_branch(line):
    source,_=archived(line,a1=False)
    step=iteration.line_index(line)['steps'][0]
    assert not any(n['id'].endswith('/main@4') for n in step['nodes'])
    assert any(n['id'].endswith('/main/2-1@4') and n['batch']==source.name for n in step['nodes'])
    assert not any(e.get('start')==2 and e.get('end')==4 and e['source'].endswith('/main@2') for e in step['edges'])


def test_resumed_branch_can_exist_only_in_outcomes_without_new_concrete_frames(line):
    source,_=archived(line,a1=False)
    tree=load_document(source/'calls/0.json.gz')
    tree['frames']=[f for f in tree['frames'] if f['branch']!='main/2-1']
    save_document(source/'calls/0.json.gz',tree)
    step=iteration.line_index(line)['steps'][0]
    assert any(n['id'].endswith('/main/2-1@0') for n in step['nodes'])
    assert any(e['category']=='rollback' for e in step['edges'])


def test_media_sampling_preserves_source_start_end_and_does_not_edit_recording(line):
    source=batch(line/'20260101-120000-a')
    original=(source/'trajectories.hdf5').read_bytes()
    index=iteration.line_index(line)
    assert sum(s['kind']=='still' for s in index['media'])==2 and sum(s['kind']=='clip' for s in index['media'])==1
    clip=dict(next(s for s in index['media'] if s['kind']=='clip'),stride=3)
    [prepared]=iteration.prepare_media(line,[clip],line.parent/'media')
    assert prepared['controls']==[0,3,4]
    with h5py.File(line.parent/'media/trajectories.hdf5') as f:
        group=f[f'data/demo_{prepared["seed"]}']
        assert group['initial_state/x'][()].tolist()==[[10]]
        assert group['states/x'][()].tolist()==[[13],[14],[14]]
        assert group.attrs['num_samples']==3
    assert (source/'trajectories.hdf5').read_bytes()==original
    assert iteration.read(line.parent/'media/media.json')['cache']==str((line/'.iteration').resolve())


def test_partial_publication_never_exposes_an_incomplete_renderer_video(tmp_path,monkeypatch):
    batch=tmp_path/'media';(batch/'videos/0').mkdir(parents=True)
    (batch/'videos/0/merge.mp4').write_bytes(b'incomplete video')
    iteration.write(batch/'media.json',dict(cache=str(tmp_path/'.iteration'),
        spans=[dict(seed=0,kind='still',path='media/node.jpg',stride=5)]))
    def incomplete(command,**kwargs):
        raise iteration.subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(iteration.subprocess,'run',incomplete)
    assert iteration.finish_media(batch,ready_only=True)==0
    assert list((tmp_path/'.iteration/media').iterdir())==[]


def test_site_lists_lines_indexes_on_demand_and_serves_steps(line,monkeypatch):
    batch(line/'20260101-120000-a')
    older=line.parents[2]/'000002'/'x1pro'/'0000'
    batch(older/'20261030-120000-z')
    save_document(older/'20261030-120000-z'/'calls'/'0.json.gz',dict(version=1,frames=[]))  # recorded before #114
    runs=line.parents[2]
    web=site.Site(runs)
    assert list(web.lines())==['000002/x1pro/0000','000100/x1pro/0000']
    assert [web.browsable(d) for d in web.lines().values()]==[False,True]
    assert [iteration.latest(d) for d in web.lines().values()]==['20261030-120000','20260101-120000']
    with pytest.raises(ValueError,match='#114'):
        web.index('000002/x1pro/0000')  # listed, never indexed
    batch(older/'20261101-120000-y')  # a recorded run after #114 makes the line browsable; earlier runs are counted
    while web.building or web.index('000002/x1pro/0000')[1]:
        threading.Event().wait(.05)
    assert web.summary('000002/x1pro/0000')['before_facts']==['20261030-120000-z']
    assert web.settings()=={'auto_render':False}
    index,state=web.index('000100/x1pro/0000')
    assert state['state']=='building'
    while web.building:
        threading.Event().wait(.05)
    summary=web.summary('000100/x1pro/0000')
    assert summary['build'] is None and summary['steps'][0]['key']=='s01'
    assert summary['steps'][0]['media']==0 and summary['steps'][0]['media_total']==3
    assert summary['steps'][0]['failures']==summary['failures']=={'a1':1}
    step=web.step('000100/x1pro/0000','s01')
    assert len(step['trees'])==1 and step['trees'][0]['nodes'][0]['media_ready'] is False
    started=[]
    monkeypatch.setattr(site.threading,'Thread',lambda target,args,daemon:type('T',(),dict(start=lambda self:started.append(args)))())
    job=web.render('000100/x1pro/0000','item',media=[step['trees'][0]['nodes'][0]['media']])
    assert job['stills']==1 and job['clips']==0 and len(started)==1
    assert web.render('000100/x1pro/0000','item',media=job['media']) is None  # already queued
    assert web.update_settings({'auto_render':True,'other':1})=={'auto_render':True}
    assert site.Site(runs).settings()['auto_render'] is True


def test_site_http_routes(line):
    batch(line/'20260101-120000-a')
    iteration.line_index(line)
    handler=type('H',(site.Handler,),dict(site=site.Site(line.parents[2])))
    server=site.ThreadingHTTPServer(('127.0.0.1',0),handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        assert b'app.js' in urlopen(base+'/').read()
        assert urlopen(base+'/app.js').headers['Content-Type'].endswith('javascript')
        assert json.loads(urlopen(base+'/api/lines').read())[0]['id']=='000100/x1pro/0000'
        assert json.loads(urlopen(base+'/api/line?id=000100/x1pro/0000').read())['steps'][0]['name']=='store pen'
        step=json.loads(urlopen(base+'/api/step?line=000100/x1pro/0000&key=s01').read())
        assert step['trees'][0]['selected'] is True  # the only run reaching the last step
        request=Request(base+'/api/settings',data=b'{"auto_render": true}',headers={'Content-Type':'application/json'})
        assert json.loads(urlopen(request).read())=={'auto_render':True}
    finally:
        server.shutdown()


def forest_step(attempts, nodes, edges):
    """A step's raw graph: ``nodes`` are (branch, at, a1) and ``edges`` (branch, start, end, category)."""
    prefix='b/r0/'
    return dict(attempts=[dict(dict(batch='b',repeat=0,source_step=1,result='success',start=0,end=4,
        entry=prefix+'main@0',selected=True,commit='c'),**a) for a in attempts],
        nodes=[dict(id=f'{prefix}{b}@{at}',branch=prefix+b,at=at,batch='b',a1=[],a1_count=a1,checks=[])
               for b,at,a1 in nodes],
        edges=[dict(id=f'{prefix}{b}:{s}-{e}',source=f'{prefix}{b}@{s}',target=f'{prefix}{b}@{e}',name='move',
                    category=c,start=s,end=e) for b,s,e,c in edges],
        failures=[])


def rollback(source,target,trigger):
    return dict(id=f'rollback-{source}',source='b/r0/'+source,target='b/r0/'+target,name='RetryState.restore',
                category='rollback',trigger=trigger,start=0,end=0,failures=[])


def test_forest_merges_a_retry_without_motion_into_one_node():
    step=forest_step([dict(end=2)],[('main',0,0),('main',2,5),('main/1-1',2,3),('main/1-1/1-2',2,1)],
                     [('main',0,2,'success')])
    step['edges']+=[rollback('main@2','main/1-1@2','planning'),rollback('main/1-1@2','main/1-1/1-2@2','planning')]
    [tree]=iteration.forest(step)
    assert len(tree['nodes'])==2 and len(tree['edges'])==1
    node=next(n for n in tree['nodes'] if n['at']==2)
    assert node['a1_count']==9 and [r['moved'] for r in node['retries']]==[False,False]
    assert not any(n.get('terminal') for n in tree['nodes'])
    assert tree['chain']==[tree['edges'][0]['id']] and tree['exit']==node['id']


def test_forest_draws_a_rolled_back_branch_to_its_end_and_retries_as_a_sibling():
    step=forest_step([dict(end=4)],[('main',0,0),('main',1,0),('main',3,0),('main/1-1',1,0),('main/1-1',4,0)],
                     [('main',0,1,'success'),('main',1,3,'a3'),('main/1-1',1,4,'success'),('main',0,3,'success')])
    step['edges'].append(rollback('main@3','main/1-1@1','a3'))
    [tree]=iteration.forest(step)
    parents={e['target']:e['source'] for e in tree['edges']}
    assert len(parents)==len(tree['edges'])  # a tree: one parent each, no edge back to an ancestor
    fork='b/r0/main@1'
    assert parents['b/r0/main@3']==fork and parents['b/r0/main/1-1@4']==fork
    assert next(n for n in tree['nodes'] if n['id']=='b/r0/main@3')['terminal']=='a3'
    assert [e['target'] for e in tree['edges'] if e['id'] in tree['chain']]==[fork,'b/r0/main/1-1@4']


def test_forest_has_one_tree_per_attempt_in_time_order_and_reads_b_from_the_classifier():
    step=forest_step([dict(batch='20260102-000000-x',entry='late/r0/main@0',selected=True),
                      dict(batch='20260101-000000-y',entry='b/r0/main@0',selected=False,result='pending')],
                     [('main',0,0)],[])
    step['nodes'][0]['checks']=[dict(category='b')]
    step['nodes'].append(dict(step['nodes'][0],id='late/r0/main@0',branch='late/r0/main',checks=[]))
    first,second=iteration.forest(step)
    assert first['started']=='2026-01-01T00:00:00' and first['result']=='b' and not first['selected']
    assert second['result']=='success' and second['selected']


def test_trees_carry_the_step_commands_and_the_run_command(line):
    path=batch(line/'20260101-120000-a')
    summary=iteration.read(path/'summary.json'); summary['command']='scripts/scenes/run_expert.py 000100 --steps 1'
    iteration.write(path/'summary.json',summary)
    [tree]=iteration.forest(iteration.line_index(line)['steps'][0])
    calls=tree['calls']
    assert calls['kind']=='step' and [(c['name'],c['main'],c['enter'],c['exit']) for c in calls['children']]==[
        ('pick pen',{'obj':'pen'},0,0),('pick pen',{'obj':'pen'},0,4)]
    assert tree['run_command']=='scripts/scenes/run_expert.py 000100 --steps 1' and tree['commit']=='a'


def test_call_tree_resolves_a_callback_argument_from_the_call_below_it():
    def frame(id,parent,kind,name,arguments,control=False):
        return dict(id=id,parent=parent,kind=kind,name=name,arguments=arguments,enter=0,exit=4,branch='main',control=control)
    callback={'$':'callback','name':'inside_region.<locals>.where'}
    frames=[frame(0,None,'step','put pen away',{'index':1}),
            frame(1,0,'skill','chain',{'actions':[{'$':'Pick','value':{'obj':'pen'}},{'$':'Place','value':{'obj':'pen','region':callback}}]}),
            frame(2,1,'subtask','put away the pen',{}),
            frame(3,2,'skill','Place',{'action':{'$':'Place','value':{'obj':'pen','region':callback}}}),
            frame(4,3,'skill','retry',{'attempts':3},control=True),
            frame(5,4,'skill','place',{'obj':'pen','region':[[0.9,0.1],[1.0,0.2]],'arm':'right'})]
    calls=iteration._call_tree(frames,frames[0])
    [chain]=calls['children']
    assert chain['main']['actions']=='Pick(pen) → Place(pen)'
    place=chain['children'][0]['children'][0]
    assert place['main']=={'obj':'pen','region':callback} and place['resolved']=={'region':[[0.9,0.1],[1.0,0.2]]}
    [retry]=place['children']
    assert retry['control'] and retry['children'][0]['main']=={'obj':'pen','arm':'right','region':[[0.9,0.1],[1.0,0.2]]}


@pytest.mark.parametrize('controls', [4, 80])
def test_unrecorded_repeat_keeps_its_tree_without_media_or_reading_main_states(line, monkeypatch, controls):
    source = batch(line/'20260101-120000-a')
    batch(source, repeat=1)
    report = iteration.read(source/'episodes/0-1.json')
    report['control_steps'] = controls
    iteration.write(source/'episodes/0-1.json', report)
    tree = load_document(source/report['call_trace'])
    for entry in tree['frames']:
        if entry['exit']:
            entry['exit'] = controls
    tree['outcomes'][0]['at'] = controls
    save_document(source/report['call_trace'], tree)
    index = iteration.line_index(line)
    assert all(span['repeat'] == 0 for span in index['media'])
    step = index['steps'][0]
    assert {attempt['repeat'] for attempt in step['attempts']} == {0, 1}
    repeat_nodes = [node for node in step['nodes'] if node['repeat'] == 1]
    repeat_edges = [edge for edge in step['edges'] if edge.get('repeat') == 1]
    assert repeat_nodes and repeat_edges
    assert all(item['media'] is None and item['media_unavailable'] == '未录制轨迹' for item in [*repeat_nodes, *repeat_edges])
    repeat_forest = next(tree for tree in iteration.forest(step) if tree['repeat'] == 1)
    assert all(item['media_unavailable'] == '未录制轨迹' for item in [*repeat_forest['nodes'], *repeat_forest['edges']])
    # A stale caller-supplied span must also be filtered before opening HDF5.
    stale = dict(batch=source.name, repeat=1, report_file='episodes/0-1.json', kind='still',
                 start=controls, end=controls, stride=5, path='media/unrecorded.jpg', step='s1')
    original_file = h5py.File
    def no_main_read(path, *args, **kwargs):
        assert Path(path).resolve() != (source/'trajectories.hdf5').resolve()
        return original_file(path, *args, **kwargs)
    monkeypatch.setattr(h5py, 'File', no_main_read)
    assert iteration.prepare_media(line, [stale], line.parent/'media') == []
    assert iteration.read(line.parent/'media/media.json')['spans'] == []
