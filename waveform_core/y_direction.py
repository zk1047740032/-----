"""Single-record direction proposals from a saved local response calibration.

No runtime neighbour captures, inverse parameter solve, GUI or device IO.
Finite-difference responses are a provisional model, not verified control gains.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


MODEL_PATH = Path(__file__).with_name('data')/'y_direction_v1.json'


def load_direction_model(path=MODEL_PATH):
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    if model.get('schema') != 'local-y-response-v1' or model.get('step') != 10:
        raise ValueError('方向模型格式或固定10步进不匹配')
    return model


try:
    DEFAULT_MODEL = load_direction_model()
except (OSError, ValueError):
    DEFAULT_MODEL = None


def waveform_fingerprint(time_s, voltage_v):
    digest = hashlib.sha256()
    for values in (time_s, voltage_v):
        array = np.asarray(values, dtype='<f8')
        digest.update(len(array).to_bytes(8, 'little'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _normalized_pulse(pulse):
    phase = np.asarray(pulse['time_ns'])/pulse['width50_ns']
    voltage = np.asarray(pulse['voltage_v'])-pulse['baseline_v']
    body = voltage[(phase >= 0) & (phase <= 1)]
    mean = float(np.mean(body))
    if len(body) < 24 or not np.isfinite(mean) or mean <= 0:
        raise ValueError('完整W50范围的有效样本不足或扣基线均值无效')
    return phase, voltage/mean


def build_direction_model(captures, center_parameters, *, known_captures=(), calibration_steps=(10,20,30)):
    """Build once from explicit single-axis captures; extra known IDs add no fit data.

    Each capture contributes one mean profile regardless of its pulse count.
    The three perturbation amounts describe this calibration experiment only.
    They are not runtime adjustment sizes, acceptance limits or repeat counts.
    """
    captures = list(captures)
    center = np.asarray(center_parameters, float)
    if center.shape != (3,) or not np.isfinite(center).all() or not captures:
        raise ValueError('需要明确的三个中心参数及校准采集')
    for kind in ('capture_id','source_sha256','waveform_sha256','waveform_fingerprint'):
        identities = [getattr(c.record,kind,None) if kind=='waveform_fingerprint' else getattr(c,kind,None) for c in captures]
        identities = [value for value in identities if value]
        if len(identities)!=len(set(identities)):
            raise ValueError(f'校准采集身份重复({kind})，不能当作独立数据拟合')
    points = {tuple(c.operating_point) for c in captures}
    if len(points) != 1:
        raise ValueError('一个局部方向模型只能校准一个工况')
    grid = np.linspace(0., 1., 201)
    profiles, catalog, sources = {}, {}, []
    for capture in captures+list(known_captures):
        if capture.error or capture.record is None or capture.record.errors or not capture.record.pulses:
            raise ValueError('校准/身份来源存在读取或定位错误，不能静默丢弃')
        identity = getattr(capture.record, 'waveform_fingerprint', None)
        metadata = dict(operating_point=list(capture.operating_point), parameters=list(capture.parameters))
        if identity:
            if identity in catalog and catalog[identity] != metadata:
                raise ValueError('同一波形身份对应不同工况或参数，需先核对')
            catalog[identity] = metadata
    for capture in captures:
        pulses = []
        for pulse in capture.record.pulses:
            phase, voltage = _normalized_pulse(pulse)
            if phase[0] > 0 or phase[-1] < 1:
                raise ValueError('校准脉冲缺少完整W50范围')
            pulses.append(np.interp(grid, phase, voltage))
        profiles.setdefault(tuple(capture.parameters), []).append(np.mean(pulses, axis=0))
        sources.append(dict(capture_id=capture.capture_id, source_sha256=capture.source_sha256,
                            parameters=list(capture.parameters), operating_point=list(capture.operating_point)))
    means = {key: np.mean(values, axis=0) for key,values in profiles.items()}
    if tuple(center) not in means:
        raise ValueError('局部响应校准需要已测中心波形')
    responses = {}
    local_profiles = {}
    for axis in range(3):
        templates = []
        for amount in calibration_steps:
            plus, minus = center.copy(), center.copy()
            plus[axis] += amount; minus[axis] -= amount
            if tuple(plus) not in means or tuple(minus) not in means:
                continue
            templates.append(dict(calibration_delta=amount,
                response_per_step=((means[tuple(plus)]-means[tuple(minus)])/(2*amount/10)).tolist()))
        responses[f'y{axis}'] = templates
        neighbours = []
        for values,profile in means.items():
            offset = np.asarray(values)-center
            if offset[axis] != 0 and np.count_nonzero(offset)==1:
                neighbours.append(dict(delta=float(offset[axis]), profile=profile.tolist()))
        local_profiles[f'y{axis}'] = sorted(neighbours,key=lambda item:item['delta'])
    parameters = np.asarray([c.parameters for c in captures], float)
    return dict(schema='local-y-response-v1', step=10, operating_point=list(next(iter(points))),
                center_parameters=center.tolist(), calibration_parameter_box=np.c_[parameters.min(0),parameters.max(0)].tolist(),
                phase_grid=grid.tolist(), responses=responses, center_profile=means[tuple(center)].tolist(),
                local_profiles=local_profiles, source_catalog=catalog, calibration_sources=sources,
                identity_catalog_used_for_fitting=False,
                normalization='own baseline-subtracted mean over rising50-to-falling50',
                scope='local one-factor calibration only; joint moves, new conditions and hardware effects unvalidated',
                hardware_bounds_confirmed=False, hardware_loop_verified=False)


def recommend_y_direction(record, phase_window, *, operating_point=None, parameters=None, model=None):
    """One prepared record + saved model -> per-axis advice and at most one trial.

    A proposal requires all available calibration responses to favour the same
    one-step direction and to have compatible shape orientation in this window.
    A disagreement is uncertainty, never disguised as hold. No RMS pass limit.
    """
    model = DEFAULT_MODEL if model is None else model
    report = dict(schema='single-wave-y-v1', status='uncertain', step=10,
        directions={f'y{i}':'uncertain' for i in range(3)}, recommendation=None,
        operating_point=None, parameters=None, context_source=None,
        per_parameter={}, pulses=[], model_schema=model.get('schema') if model else None,
        model_operating_point=model.get('operating_point') if model else None,
        direction_validated=False, device_command_allowed=False, rms_acceptance_limit=None,
        uses_runtime_comparison_captures=False, reason='')

    def unavailable(reason, status='uncertain'):
        report.update(status=status, reason=reason)
        return report

    if model is None:
        return unavailable('内置响应模型不可用；评分仍可使用。', 'model_unavailable')
    context = dict(getattr(record, 'measurement_context', None) or {})
    saved = model.get('source_catalog', {}).get(getattr(record, 'waveform_fingerprint', None))
    if saved:
        context = dict(saved, **context)
        report['context_source'] = 'saved_waveform_identity'
    if operating_point is not None:
        context['operating_point'] = list(operating_point)
        report['context_source'] = 'explicit_metadata'
    if parameters is not None:
        context['parameters'] = list(parameters)
        report['context_source'] = 'explicit_metadata'
    if report['context_source'] is None and context:
        report['context_source'] = 'record_metadata'
    report.update(operating_point=context.get('operating_point'), parameters=context.get('parameters'))
    checked = {}
    for key,size in (('operating_point',2),('parameters',3)):
        if context.get(key) is None:
            continue
        try:
            values = np.asarray(context[key],float)
            if values.shape!=(size,) or not np.isfinite(values).all() or (key=='operating_point' and min(values)<=0):
                raise ValueError('invalid metadata')
        except (ValueError,TypeError):
            report.update(operating_point=None,parameters=None,invalid_context_input=repr(context))
            return unavailable('当前工况或参数元数据无效。', 'invalid_context')
        checked[key] = values
        report[key] = values.tolist()
    if context.get('operating_point') is None or context.get('parameters') is None:
        return unavailable('需确认当前频率、脉宽和y参数；不需要当前波形的±10配套文件。', 'context_required')
    point,current = checked['operating_point'],checked['parameters']
    if not np.array_equal(point, model['operating_point']):
        return unavailable('当前工况不在内置模型已校准工况内，不能套用其他工况方向。', 'out_of_scope')
    box = np.asarray(model['calibration_parameter_box'])
    if np.any(current < box[:,0]) or np.any(current > box[:,1]):
        return unavailable('当前y超出模型已有校准范围；这不是设备安全限值。', 'out_of_scope')
    if record.errors or not record.pulses:
        return unavailable('本条波形不能完整评价。', 'invalid_waveform')
    left,right = phase_window
    if not np.isfinite([left,right]).all() or not 0 <= left < right <= 1:
        return unavailable('保持期窗口尚未有效确认。', 'window_confirmation_required')
    grid = np.asarray(model['phase_grid'])

    def rms(values):
        mean = float(np.mean(values))
        if mean <= 0:
            return float('inf')
        blocks = np.array([part.mean() for part in np.array_split(values,12)])
        return float(np.sqrt(np.mean((blocks-mean)**2))/mean*100)

    try:
        for pulse in record.pulses:
            phase, voltage = _normalized_pulse(pulse)
            mask = (phase >= left) & (phase <= right)
            tx,z = phase[mask], voltage[mask]
            if len(z) < 24 or not np.isfinite(z).all():
                raise ValueError('保持期有效样本不足')
            score = rms(z)
            axes = {}
            for axis in range(3):
                key = f'y{axis}'
                templates = model['responses'].get(key, [])
                vectors, votes, gains = [], [], []
                for template in templates:
                    response = np.interp(tx, grid, template['response_per_step'])
                    vector = np.array([part.mean() for part in np.array_split(response,12)])
                    vectors.append(vector-vector.mean())
                    plus,minus = rms(z+response),rms(z-response)
                    direction = ('increase' if plus < min(score,minus) else 'decrease' if minus < min(score,plus)
                                 else 'hold' if min(plus,minus) >= score else 'uncertain')
                    votes.append(dict(calibration_delta=template['calibration_delta'], direction=direction))
                    gains.append(score-min(plus,minus))
                compatible = (len(vectors) >= 2 and all(np.dot(a,b)>0 for i,a in enumerate(vectors) for b in vectors[i+1:]))
                labels = {vote['direction'] for vote in votes}
                direction = next(iter(labels)) if compatible and len(labels)==1 else 'uncertain'
                local = model['local_profiles'][key]+[dict(delta=0.,profile=model['center_profile'])]
                local = sorted(local,key=lambda item:item['delta'])
                knots = np.array([item['delta'] for item in local])
                profiles = np.asarray([item['profile'] for item in local])
                offset = current[axis]-model['center_parameters'][axis]
                def profile_at(value):
                    return np.array([np.interp(value,knots,column) for column in profiles.T])
                local_current = profile_at(offset)
                local_scores = {}
                for sign in (1,-1):
                    if knots[0] <= offset+sign*10 <= knots[-1]:
                        change = profile_at(offset+sign*10)-local_current
                        local_scores[sign] = rms(z+np.interp(tx,grid,change))
                plus,minus = local_scores.get(1,float('inf')),local_scores.get(-1,float('inf'))
                local_vote = ('increase' if plus < min(score,minus) else 'decrease' if minus < min(score,plus)
                              else 'hold' if local_scores and min(plus,minus)>=score else 'uncertain')
                local_disagrees = direction != 'uncertain' and direction != local_vote
                if local_disagrees:
                    direction = 'uncertain'
                if score == 0:
                    direction = 'hold'  # exact zero is a mathematical minimum, not an RMS pass threshold
                reason = ('不同校准响应在当前窗口内方向不一致。' if not compatible else
                          '不同校准响应的一步建议不一致。' if len(labels)!=1 else
                          '模型在本窗口未见该参数一步改善；不表示整体合格。' if direction=='hold' else
                          '多个局部响应模型给出同向试调建议，尚未实机验证。')
                if score == 0:
                    reason = '当前分块RMS恰为零，无需由本模型调整；不构成整体验收结论。'
                elif local_disagrees:
                    reason = '跨幅度响应与当前y附近的一步响应不一致，不据此给确定方向。'
                if direction in ('increase','decrease'):
                    trial = current[axis]+(10 if direction=='increase' else -10)
                    if not box[axis,0] <= trial <= box[axis,1]:
                        direction,reason = 'uncertain','下一步超出已有模型校准范围，不能据此继续外推。'
                model_gains = gains+[score-min(plus,minus)]
                worst_gain = float(min(model_gains)) if model_gains and np.isfinite(model_gains).all() else None
                if worst_gain is None and direction in ('increase','decrease'):
                    direction,reason = 'uncertain','响应模型无法给出有限的一步比较。'
                axes[key] = dict(direction=direction, reason=reason, response_orientation_compatible=compatible,
                                 votes=votes, local_step_vote=local_vote, local_parameter_offset=float(offset),
                                 worst_model_improvement_pp=worst_gain,
                                 proxy_only=True)
            report['pulses'].append(dict(number=pulse['number'], per_parameter=axes))
    except (ValueError, FloatingPointError) as error:
        report['pulses'] = []
        return unavailable(str(error), 'invalid_waveform')
    for axis in range(3):
        key = f'y{axis}'
        rows = [pulse['per_parameter'][key] for pulse in report['pulses']]
        labels = {row['direction'] for row in rows}
        direction = next(iter(labels)) if len(labels)==1 else 'uncertain'
        report['directions'][key] = direction
        report['per_parameter'][key] = dict(direction=direction,
            reason=rows[0]['reason'] if len(labels)==1 else '记录内脉冲给出不同方向，不合并为确定建议。')
    choices = [(min(p['per_parameter'][key]['worst_model_improvement_pp'] for p in report['pulses']), key, direction)
               for key,direction in report['directions'].items() if direction in ('increase','decrease')]
    if choices:
        _,key,direction = max(choices)
        report.update(status='exploratory', recommendation=dict(parameter=key,direction=direction,step=10),
                      reason='仅建议先试一个y的10步进；改后必须用新采集验证，未证明复采改善。')
    elif set(report['directions'].values()) == {'hold'}:
        report.update(status='hold', reason='本模型未见当前波形的单步改善方向，不代表整体达标或已最优。')
    else:
        report['reason'] = '现有模型不能支持一致的下一步；保留不确定，不要求配齐六方向文件。'
    return report
