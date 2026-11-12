# F2A2-Isaac

Isaac Lab에서 두 ANYmal C가 같은 막대를 지지하며 이동·회전 명령을 추종하는 **Collaborative Bar Balancing** 과제에 [F2A2-SAC](https://arxiv.org/abs/2004.11145v2)을 적용하는 연구 프로젝트입니다. 원본 태스크 ID는 `Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0`입니다.

로봇마다 독립 actor가 자기 관측 48차원으로 행동 12차원을 생성합니다. 학습에는 상대 관측·행동과 critic 가중치 snapshot을 명시적 텐서 복사로 전달합니다. 각 learner가 자기 Replay, critic, target critic, MOA, optimizer와 PDHG 상태를 관리합니다.

현재는 **실제 ANYmal에서 데이터 수집, MOA 갱신, Twin-Q TD 업데이트, PDHG 합의까지** 짧게 실행할 수 있습니다. Proposition 6의 joint actor–critic 업데이트와 장기 학습 루프는 개발 중입니다.

## 어떤 파일을 실행하면 되나요?

| 확인할 내용 | 실행할 파일과 옵션 | 필요한 환경 |
| --- | --- | --- |
| 관측·행동 수집, 두 Replay에 저장, importance ratio 계산 | `python smoke_collect.py` | Python, PyTorch |
| critic 복사본 교환과 PDHG 합의 | `python learner.py` | Python, PyTorch |
| Replay 이력 구성과 MOA 학습 | `python smoke_moa.py` | Python, PyTorch |
| 실제 ANYmal 환경에서 수집하고 Twin-Q 출력 확인 | `python collect_isaac.py`와 아래 태스크 인자 | Isaac Sim, Isaac Lab, IsaacLab-HARL 태스크 소스 |
| 실제 수집부터 TD 업데이트·PDHG·MOA까지 확인 | `python collect_isaac.py --td-smoke`와 아래 태스크 인자 | Isaac Sim, Isaac Lab, IsaacLab-HARL 태스크 소스 |

## PyTorch로 빠르게 확인하기

저장소 루트에서 아래 명령을 실행합니다. 세 검사는 CPU에서 실행할 수 있습니다.

```bash
python smoke_collect.py
OMP_NUM_THREADS=2 python learner.py
OMP_NUM_THREADS=2 python smoke_moa.py
```

- `smoke_collect.py`는 두 로봇에 각각 `replay=12`와 `mean ratio=1.000`을 출력합니다. 수집 뒤 actor를 업데이트하지 않았기 때문에 현재 정책과 수집 정책의 밀도가 일치합니다.
- `learner.py`는 PDHG를 두 라운드 실행합니다. 양쪽에서 계산한 합의값이 같고 dual의 합이 0에 가까운지 확인합니다. `critic_gap`은 합의 직전 두 로컬 critic의 파라미터 거리입니다.
- `smoke_moa.py`는 각 로봇의 MOA가 작은 고정 배치에서 NLL을 낮추고, 유한한 상대 행동 샘플 `(16, 12)`를 내는지 확인합니다.

## Isaac Lab에서 실행하기

Isaac Sim과 Isaac Lab이 동작하는 Python 환경을 활성화하고, `--task-source`에 로컬 IsaacLab-HARL checkout의 `anymal_c_multi_agent.py`를 지정합니다. 아래 예시는 Conda 환경 이름이 `isaaclab`이고 fork가 `~/IsaacLab-HARL`에 있을 때의 명령입니다. 기존 검증에는 Python 3.11, PyTorch `2.7.0+cu128`, Isaac Sim 5.0, Isaac Lab 2.2.1(`0f00ca2b4`), 태스크 checkout `5620e5e4`를 사용했습니다.

```bash
conda activate isaaclab

HARL_DIR="$HOME/IsaacLab-HARL"
TASK_SOURCE="$HARL_DIR/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c_multi_agent/anymal_c_multi_agent.py"
```

**환경과 수집만 확인**하려면 다음 명령을 실행합니다.

```bash
python -u collect_isaac.py --headless --device cuda:0 \
  --num-envs 2 --steps 16 --task-source "$TASK_SOURCE"
```

병렬 환경 2개에서 16스텝을 실행하므로, 로봇별로 `replay=32 q1=(8,) q2=(8,)`이 출력됩니다. `final observation rows`는 자동 reset 직전의 관측을 보존한 transition 수입니다.

**실제 데이터로 critic과 MOA 업데이트까지 확인**하려면 `--td-smoke`를 추가합니다.

```bash
python -u collect_isaac.py --headless --device cuda:0 \
  --num-envs 2 --steps 16 --task-source "$TASK_SOURCE" --td-smoke
```

위 설정에서는 먼저 MOA의 고정 Character 궤적을 위해 로봇별 50건을 수집합니다. 별도 수집에서 얻은 로봇별 32건으로 TD critic 업데이트, critic snapshot 교환과 합의, target 및 MOA 갱신을 두 라운드 실행합니다. 정상적으로 완료되면 `round=0`과 `round=1`에 두 로봇의 `target_mean`과 `td_mse`가 각각 출력됩니다.

## 파일별 역할

| 파일 | 역할 |
| --- | --- |
| [`collect_isaac.py`](collect_isaac.py) | Isaac Sim을 시작하고 로컬 HARL의 막대 균형 태스크를 불러옵니다. 두 로봇의 행동으로 환경을 진행하고 각 Replay에 저장합니다. `terminated`와 `truncated`의 반환 순서를 맞추고 자동 reset 직전 관측을 따로 보존합니다. 실제 환경의 실행 진입점입니다. |
| [`actor.py`](actor.py) | 자기 관측 48차원으로 12차원 대각 Gaussian 정책을 구성합니다. `tanh` 행동을 샘플링하고, Replay에 저장할 수집 당시 분포 파라미터와 `tanh` 변환을 보정한 로그 밀도를 계산합니다. |
| [`replay.py`](replay.py) | 각 learner의 transition을 저장하는 링 버퍼입니다. 상대 데이터는 `copy_for_peer()`로 복사하고, 환경·episode·시점 ID를 기준으로 MOA의 과거 이력과 고정 Character 궤적을 구성합니다. |
| [`critic.py`](critic.py) | 관측·행동을 `robot_0`, `robot_1` 순서로 조립해 두 attention Q 모델에 입력합니다. 각 learner는 자기 로봇의 `Q1`, `Q2` 출력을 사용합니다. |
| [`moa.py`](moa.py) | 상대의 현재 관측, 현재 episode의 과거 이력, 별도로 수집한 Character 궤적으로 상대의 12차원 행동 분포를 추정합니다. 다음 관측에서의 상대 행동 샘플은 TD target에 사용합니다. |
| [`consensus.py`](consensus.py) | critic 파라미터의 독립 snapshot을 만들고, 동일 라운드의 두 snapshot으로 합의값과 로컬 dual을 갱신합니다. learner가 critic 업데이트에 더할 PDHG 항도 계산합니다. |
| [`learner.py`](learner.py) | 로봇 한 대의 모델과 학습 상태를 묶습니다. TD target 계산, critic·MOA 업데이트, target critic 갱신을 담당합니다. 파일을 직접 실행하면 CPU에서 PDHG 합의를 검사합니다. |
| [`smoke_collect.py`](smoke_collect.py) | 가짜 벡터 환경에서 두 actor와 Replay를 연결해 상대 데이터 복사, 부분 reset, 저장된 정책 밀도를 확인합니다. |
| [`smoke_moa.py`](smoke_moa.py) | 별도로 수집한 가짜 rollout을 사용해 고정 Character 데이터와 Replay 이력을 구성한 뒤, 두 로봇의 MOA 학습과 행동 샘플링을 확인합니다. |
