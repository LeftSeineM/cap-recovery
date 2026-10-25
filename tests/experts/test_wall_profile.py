"""Older reports and candidates absent in some episodes must not bias category means."""

from scripts.experts.step_times import profile_times, step_profiles


def row(seconds):
    return {'calls': 1, 'wall_s': seconds, 'self_s': seconds}


def test_means_include_zero_for_absent_searches_and_ignore_unprofiled_reports():
    assert profile_times([{}, {'wall_profile': {'handover': row(4.)}}, {'wall_profile': {}}]) == {
        'handover': {'calls': .5, 'wall_s': 2., 'self_s': 2.}}
    assert profile_times([{}]) == {}


def test_step_profile_deltas_and_old_reports():
    report = {'steps': [
        {'step': 2, 'name': 'pick', 'wall_profile': {'ik': row(3.)}},
        {'step': 3, 'name': 'put', 'wall_profile': {'ik': row(5.), 'rrt': row(2.)}},
    ]}
    rows = list(step_profiles(report))
    assert rows[0] == ((2, 'pick'), {'ik': row(3.)})
    assert rows[1][1]['ik'] == {'calls': 0, 'wall_s': 2., 'self_s': 2.}
    assert rows[1][1]['rrt'] == row(2.)
    assert list(step_profiles({'steps': [{'step': 1, 'name': 'old'}]})) == []


def test_cpu_and_instruction_means_preserve_missing_measurements():
    old = {'wall_profile': {'ik': row(20.)}}
    cpu = {'wall_profile': {'ik': {**row(4.), 'thread_cpu_s': 2., 'self_thread_cpu_s': 1.}}}
    hardware = {'wall_profile': {'ik': {**row(6.), 'thread_cpu_s': 4., 'self_thread_cpu_s': 3.,
                                       'instructions': 100, 'self_instructions': 80}}}
    absent = {'wall_profile': {'rrt': {**row(2.), 'thread_cpu_s': 2., 'self_thread_cpu_s': 2.}}}
    result = profile_times([old, cpu, hardware, absent])['ik']
    assert result['self_s'] == 7.5
    assert result['self_thread_cpu_s'] == 4. / 3.
    assert result['self_instructions'] == 80
