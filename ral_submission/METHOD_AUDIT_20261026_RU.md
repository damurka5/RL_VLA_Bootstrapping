# Математический аудит loss и расписания обучения

Дата: 26 октября 2026. Проверен код на commit 0b2d5cd и история Git. Обучение и запуск робота не выполнялись. Исторические resolved configs и веса всех запусков не доступны в этом аудите; наличие опции в коде отдельно от её включения в выбранной конфигурации.

## 1. Главный результат и границы описания

Базовый RL objective — PPO-style clipped surrogate с групповыми GRPO advantages. Существенные изменения находятся в параметризации действий, распределении exploration, вычислении likelihood, распределении credit по сегментам и взвешивании loss.

В репозитории есть две разные реализации композиции:

| Реализация | Actor и likelihood | Credit / reward | LoRA |
|---|---|---|---|
| CDPR, вариант с результатом 61.43% | bounded_residual_v0 + latent_gaussian_conditional_offset_v1 | три независимых milestone advantages, terminal bonus, защита достигнутого префикса | сохранена и заморожена |
| Текущий автоматический WX200 pipeline | bounded_residual_v0 + clipped_action_v0 по умолчанию: YAML не переопределяет action_likelihood | staged_milestone_reward: доля достигнутых milestones, один return на инструкцию | train_vla_lora=true, updates включены по умолчанию |

Семистадийное расписание и численные early-stop пороги ниже относятся к текущему WX200 pipeline. Они не устанавливают критерии исторических ручных переходов в CDPR.

Источники: [actor и default flags](../rl_vla_bootstrapping/policy/smolvla_grpo_finetune_cdpr.py), [CDPR launcher](../scripts/train_cdpr_latent_correction_pilot_remote.sh), [WX200 config](../configs/examples/widowx200_smolvla_pipeline.yaml), [виды reward](../rl_vla_bootstrapping/simulation/cdpr_batched_tasks.py).

## 2. Параметризация действий: bounded residual

Обозначим p — нормализованный action prior SmolVLA, x — 518d наблюдение: 512 визуальных признаков и 6 значений proprioception. Сеть также получает flatten(p), то есть 40 значений для 8×5 chunk.

$$z=[x,\operatorname{vec}(p)],\qquad
d_\theta=s_r\tanh f_\theta(z),\qquad
\mu_\theta=\tanh(p+d_\theta).$$

В проверенных основных конфигах s_r=1, MLP: 558 → 1024 → 1024 → 40. Предсказывается 8 слотов, выполняются 4, затем prior и chunk пересчитываются.

Обе tanh имеют значение:

- внутренняя ограничивает residual: |d_j| ≤ s_r;
- внешняя ограничивает среднюю команду;
- при фиксированном prior детерминированное среднее ограничено интервалом [tanh(p_j−s_r), tanh(p_j+s_r)];
- производная по выходу сети равна s_r(1−mu_j²)(1−tanh²(f_j)). Это ограничивает чувствительность action mapping, но не гарантирует ограничение полного градиента по параметрам сети.

Начальный residual создавался обычной инициализацией Linear, не нулями. Поэтому при scratch-start не обеспечивается mu=p или точное сохранение исходного поведения. Даже при f=0 получается mu=tanh(p).

Источник: [ResidualChunkActor](../rl_vla_bootstrapping/policy/octo_finetune_cdpr.py), строки 785–817.

## 3. Самое важное изменение для RL: что именно получает log-probability

После вычисления среднего добавляются случайные возмущения:

$$B\sim q(B),\quad b_t=g_t B,\quad
u_t=\mu_\theta(x_t,p_t)+b_t+\sigma_\theta\odot\epsilon_t,\quad
a_t=\operatorname{clip}(u_t,-1,1),\quad \epsilon_t\sim\mathcal N(0,I).$$

B сохраняется на протяжении эпизода; gate g_t разрешает возмущение на нужном сегменте. В CDPR three-stage режиме gripper offset включается после held lift, чтобы exploration release не открывал пальцы во время подхода/захвата.

Важно: tanh применяется к среднему ДО Gaussian noise. Это не распределение a=tanh(z), где Gaussian latent целиком проходит через tanh. Для данного sampling path не нужен Jacobian squashed Gaussian.

### 3.1. Исторический путь

В clipped_action_v0 Gaussian log-density вычислялась в выполненном, уже обрезанном действии a_t. После появления episode offset использовалась расширенная дисперсия sigma²+eta².

У распределения clipped Gaussian есть атомы на границах. Например, в одной координате:

$$\Pr(a_t=1)=1-\Phi((1-\mu_t-b_t)/\sigma_t).$$

Эта вероятность не равна значению Gaussian density в точке 1. Вдобавок повторяющийся episode offset создаёт временную зависимость; произведение независимых расширенных marginal densities её не описывает.

### 3.2. Исправленный путь

Новая ветка хранит раздельно executed_action, policy_sample=u, behavior_mean_offset=b и old_log_prob. При обновлении:

$$\ell_{\theta,t}=
-\frac12\sum_j\left[
\frac{(u_{tj}-\mu_{\theta,tj}-b_{tj})^2}{\sigma_{\theta,tj}^2}
+2\log\sigma_{\theta,tj}+\log(2\pi)\right].$$

Для фиксированного sigma mean-score:

$$\nabla_\theta\ell_{\theta,t}
=(\nabla_\theta\mu_{\theta,t})^\top
\operatorname{diag}(\sigma_\theta^{-2})(u_t-\mu_{\theta,t}-b_t).$$

Внутри допустимого диапазона log_std её Gaussian score в координате j равен (u_j−mu_j−b_j)²/sigma_j²−1. Recorded u и b используются как фиксированные данные при differentiation.

Среда исполняет a, но policy gradient оценивает породивший его latent u. Clipping становится детерминированным преобразованием среды. Плотность независимого от параметров B сокращается в отношении trajectory likelihoods. Это корректная likelihood-модель; этапные фильтры, нормализация advantage и PPO clipping по-прежнему меняют полный оптимизируемый surrogate.

Источник: [sampler](../rl_vla_bootstrapping/policy/smolvla_grpo_finetune_cdpr.py), строки 1468–1548 и 1680–1695; [latent density](../rl_vla_bootstrapping/policy/latent_correction_policy.py), строки 101–120. Изменение добавлено commit b228e3a, 3 октября 2026; ранние legacy-запуски автоматически на него не переводились.

## 4. Полный actor loss

Для записи r, соответствующей одному выполненному 5d slot:

$$\rho_r=\exp\{\operatorname{clip}(\ell_{\theta,r}-\ell_{\mathrm{old},r},-20,20)\},$$
$$\mathcal S_r(\theta)=\min\{\rho_r\widetilde A_r,\,
\operatorname{clip}(\rho_r,1-\epsilon_-,1+\epsilon_+)\widetilde A_r\}.$$

Численная отсечка log-ratio ±20 и PPO clipping — разные операции. В основных SmolVLA конфигах epsilon_−=0.20 и epsilon_+=0.28: ratio ограничивается диапазоном [0.80,1.28] во второй ветви surrogate.

$$\mathcal L_{\mathrm{actor}}=
-\sum_r w_r\mathcal S_r(\theta)
-c_H\sum_r w_r H[\mathcal N(\mu_{\theta,r},\Sigma_{\theta,r})]
+\lambda_a\sum_r w_r\frac{\|\mu_{\theta,r}\|_2^2}{d}.$$

Веса здесь обозначают нормировку реализованного record/stage objective. В обычном режиме используется среднее по пригодным записям, в three-stage — отдельная нормировка по кандидатам и стадиям. Ratio считается на 5d slot, не сразу на полный 40d chunk. В обычной tensor-ветке surviving advantages дополнительно центрируются и нормируются по пригодным строкам всего update. Three-stage ветка намеренно сохраняет stage/group advantages без такой повторной нормировки, чтобы не уничтожить сигнал единственного успешного downstream-кандидата.

Фактические настройки:

- entropy_coef=0 для move_to scratch; 0.0002 для проверенных pickup/placement/full-task конфигов;
- action_l2 есть в формуле и коде, но default=0; в проверенных YAML он не включён. Приписывать ему стабилизацию этих запусков нельзя;
- Adam, clipping нормы градиента до 1.0;
- log_std ограничен и после optimizer.step дополнительно проецируется в допустимый интервал.

Последнее исправляет «мёртвые» параметры за границей clamp: один forward clamp даёт нулевой градиент вне интервала. Проекция raw log_std после шага возвращает параметр на границу. В full-task конфиге диапазон [−1.72,−1.10]; scratch move_to использует другой нижний предел. Исправление: commit 80f6c35, 2 августа 2026.

Источник: [actor loss](../rl_vla_bootstrapping/policy/smolvla_grpo_finetune_cdpr.py), строки 1830–1930; project_log_std, строки 152–177.

## 5. Нестандартный credit: чему учится каждое действие

### 5.1. Две части pickup

При split_credit_at_grasp=true до захвата используется reward в момент первого захвата; после захвата — terminal reward. Если захвата не было, pre-grasp return остаётся terminal reward. Для двух потоков независимо считаются group advantages.

Так команда опускания для захвата не получает тот же return, что последующее поднятие. Стадия действия определяется ДО шага: действие, создавшее захват, относится к approach.

Источник: [collector](../rl_vla_bootstrapping/policy/mjwarp_rank_local_collector.py), строки 4877–4932. Commit a38dfb7, 5 августа 2026. В binary-фазе WX200 этот split выключен: до held lift бинарный reward был бы нулевым.

### 5.2. Три части CDPR full task

Для каждой сцены собирается группа G=8 кандидатов. Milestone returns M_ik∈{0,1}: approach, held pickup, strict placement.

$$A_{ik}=\operatorname{clip}\left(
\frac{M_{ik}-\bar M_k}{\max(\operatorname{std}_k,10^{-6})},-6,6\right).$$

Статистики считаются по валидным кандидатам, std — population standard deviation. Для usable mask U_k учитываются dynamic sampling и минимальный spread; в основных конфигах 0.1 < pass_rate < 0.9 и std ≥ 0.05. На стадиях может быть разная пригодность одной группы.

Для approach/pickup:

$$\widetilde A_{ik}=U_{ik}A_{ik}
+\beta U_{i3}
\begin{cases}
\kappa A_{i3},&M_{ik}=1,\ A_{i3}<0,\\
A_{i3},&\text{иначе}.
\end{cases}$$

Placement получает U_i3 A_i3 один раз. В проверенном full-task конфиге beta=1, kappa=0. Отрицательный terminal bonus за последующий провал не штрафует уже достигнутый prefix milestone. Положительная full-task credit передаётся ранним действиям. Группа с константным локальным milestone может стать usable через terminal contrast.

Это защита определённой компоненты credit. Shared parameters, entropy, другие записи и optimizer могут менять ранее освоенное поведение; формальной гарантии сохранения навыка нет.

Источник: three_stage_group_credit и add_full_task_bonus в [collector](../rl_vla_bootstrapping/policy/mjwarp_rank_local_collector.py), строки 246–309 и 529–585.

### 5.3. Loss нормируется по кандидатам, затем по стадиям

Схематически stage objective:

$$\mathcal L_{\mathrm{PG}}
=-\sum_{k\in\mathcal K}
\frac{\omega_k}{|\mathcal C_k|}
\sum_{i\in\mathcal C_k}\frac{1}{T_{ik}}
\sum_{t\in\mathcal T_{ik}}s_{it}(\theta).$$

Поэтому длинный неуспешный approach не перевешивает короткий успешный только из-за числа записей. Stage mass задаёт вклад стадий в loss; масштабирование binary reward отдельно не заменяет stage weights, поскольку group normalization устраняет положительный масштаб reward. В выбранном CDPR конфиге массы равны; неиспользуемые стадии исключаются из нормировки. DDP дополнительно компенсирует averaging градиентов между rank.

Источник: [global_stage_loss_weights](../rl_vla_bootstrapping/policy/rank_local_grpo.py), строки 168–234. Веса рассчитываются один раз по всему собранному update до optimizer loop, затем умножаются на число minibatches. При фиксированных параметрах случайный minibatch оценивает этот взвешенный objective в ожидании; Adam обновляет параметры после каждого minibatch.

## 6. LoRA: «KL» в названии, MSE в вычислении

Отдельный legacy LoRA update использует:

$$\mathcal L_{\mathrm{LoRA}}=
\mathcal L_{\mathrm{PPO,subsample}}
+0.1\,\operatorname{mean}\|p_\phi-p_{\mathrm{rollout}}\|_2^2.$$

Переменная называется kl, но фактическая операция — squared difference между prior chunks. Интерпретация как KL требует дополнительных допущений о распределениях и общей фиксированной covariance. Без них корректнее называть это action-prior consistency / MSE regularization.

Это ограничивает уход prior во время RL-адаптации LoRA. В latent-likelihood continuation LoRA update не поддерживается: inherited LoRA заморожена, этот loss не выполняется.

Источник: [update_vla_lora](../rl_vla_bootstrapping/policy/smolvla_grpo_finetune_cdpr.py), строки 2336–2422.

Другие опции, не смешивать с выбранным результатом:

- frozen-reference anchor: mean-term Gaussian KL на фиксированном банке, отдельная opt-in регуляризация;
- zero-init correction actor: mu=tanh(logit_ref+g_psi), последний слой g обнулён. На старте mu точно совпадает с reference, первый actor gradient идёт в выходной слой. Это экспериментальная candidate-ветка; результат 61.43% относится к bounded residual control.

## 7. Фактические training stages и критерии остановки WX200

Расписание из configs/examples/widowx200_smolvla_pipeline.yaml:

| № | Skill / reward | Метрика v | Target | Min steps, M | Budget, M | Plateau W |
|---|---|---|---:|---:|---:|---:|
| 1 | reaching / dense | success_rate | 0.70 | 3 | 12 | 10 |
| 2 | pickup / dense | success_rate | 0.55 | 3 | 16 | 12 |
| 3 | placement / dense | min(plate,bowl) | 0.70 | 2 | 12 | 10 |
| 4 | reaching / binary | success_rate | 0.80 | 1 | 6 | 8 |
| 5 | pickup / binary | success_rate | 0.65 | 1 | 8 | 8 |
| 6 | placement / binary | min(plate,bowl) | 0.80 | 1 | 6 | 8 |
| 7 | joint / staged | min всех инструкций; placement из composed leg | 0.60 | 5 | 30 | 16 |

Все dense skills идут перед всеми binary skills. Это не цикл «dense→binary» внутри каждого навыка.

Validation проводится каждые 250,000 selected environment actions. Это training counter, а не optimizer updates и не все sampled simulator interactions. Дистанции validation фиксированы и не следуют curriculum: reaching 0.30 m, pickup 0.20 m, placement 0.18 m.

Три причины завершить стадию:

1. target: v ≥ target на двух consecutive validations и достигнут min_steps;
2. plateau: n > W, достигнут min_steps и

$$\max_{n-W<j\le n}v_j-\max_{j\le n-W}v_j\le0.02;$$

3. исчерпан budget_steps.

Streak и history накапливаются также до min_steps; min_steps запрещает принять early-stop решение раньше этого счётчика. После остановки выбирается policy с максимальной stage metric на требуемых fixed-distance validations; при равенстве берётся более поздняя.

ВАЖНО: target не является обязательным допуском к следующей стадии. Plateau и budget также приводят к продолжению pipeline. Если нет ни одной подходящей fixed-distance validation, driver завершает работу с ошибкой, а не выбирает policy по несопоставимому curriculum score.

При переходе стадии: сохраняются веса policy/LoRA; optimizer и RL/curriculum state начинают заново. При входе в binary и staged log_std также сбрасывается к init_log_std. Resume прерванной той же стадии восстанавливает её состояние.

Источник: [EarlyStopRule](../rl_vla_bootstrapping/policy/smolvla_grpo_mjwarp_cdpr.py), строки 351–443; [stage driver](../tools/train/run_widowx200_pipeline.py), строки 332–360, 453–485, 565–623, 1032–1083; [early-stop metric](../rl_vla_bootstrapping/policy/early_stopping.py).

## 8. Когда запускается SIL и принимается её результат

После стадий 1–6 собирается успешный experience bank и проверяются ранее изученные инструкции. При одном изученном skill retention обычно пропускается. В joint stage выставлено sil=false.

Для reference success b_i и текущего J_i инструкция считается деградировавшей при

$$b_i-J_i>\max(0.08,\ 0.20b_i).$$

Это превышение обеих tolerances: 8 процентных пунктов и 20% reference. При деградации и доступном банке запускается до двух SIL cycles:

1. Пул успешных replay из retained skill banks. Bank заменяется только при наличии успешных replay/frames и не меньшем среднем harvest rate для skill.
2. До примерно 12,000 decision rows на instruction, с сохранением целых эпизодов.
3. Smoothing window=5, physics replay и сохранение успешных replay frames.
4. Пересчёт prior/features текущей policy на сохранённых frames; исходные instruction labels сохраняются.
5. 20 эпох masked action regression residual MLP. В текущем вызове sil_sft нет --frames, поэтому дополнительный LoRA SFT этап не запускается; LoRA tensors наследуются.
6. Оценка candidate на всех learned instructions.

$$\mathcal L_{\mathrm{SIL}}=
\frac{\sum_{r,h,j}m_{rh}\big(\mu_{\theta,rhj}-a^{\mathrm{recorded}}_{rhj}\big)^2}
{d\sum_{r,h}m_{rh}}.$$

По умолчанию это MSE; sil_sft также имеет опцию L1. Маски исключают unused slots и post-terminal tails. В отличие от RL, здесь target — записанное исполненное действие.

Candidate принимается, если средний gain на деградировавших инструкциях ≥0.02 и каждая недеградировавшая инструкция теряет не более 0.05. Проверяется среднее по degraded, а не улучшение каждой из них. Также должен существовать SFT candidate, победивший initializer по внутреннему selection score. При reject цикл прекращается; после двух неудачных/недостаточных попыток дальнейшая стадия не блокируется автоматически.

Источник: [retention and acceptance](../tools/train/run_widowx200_pipeline.py), строки 625–696, 773–831, 839–929; [masked supervised loss](../tools/audit/sil_sft.py), строки 2290–2307.

## 9. Curriculum и milestones внутри эпизода — отдельные переходы

### Curriculum сложности WX200

Для каждого instruction свой EMA успеха:

$$e_{t+1}=\gamma e_t+(1-\gamma)\hat J_t.$$

В первых skill stages cap дистанции повышается на одну ступень после пяти qualifying updates с EMA≥0.30; понижается при EMA≤0.12. Cooldown=15 updates. Обычно gamma=0.9; pickup переопределяет gamma=0.95. После cooldown отсчёт dwell продолжается согласно реализации; при изменении cap EMA пересеивается первой оценкой новой сложности.

В joint phase пороги cap — 0.45/0.30. Доля placement-стартов с уже удерживаемым объектом уменьшается от 0.5 к 0 с шагом 0.1 при EMA placement success≥0.45; восстанавливается при EMA≤0.30 до исходных 0.5. Cooldown=40, gamma=0.9. Это training-time assistance, не успех complete task.

Источник: ApproachDistanceCurriculum, PreliftedStageCurriculum и PlacementCaughtStageCurriculum в [trainer](../rl_vla_bootstrapping/policy/smolvla_grpo_mjwarp_cdpr.py).

### Task milestones CDPR

Текущий advance_three_stage_milestones использует:

- approach: расстояние до grasp point ≤0.03 m ИЛИ реальный persistent grasp;
- pickup: physical/task grasp и held lift ≥0.05 m;
- placement: strict full-task outcome, включая deliberate release и отсутствие latched carry-slip / wrong placement.

Достигнутые milestones latch-ятся. Strict success принудительно подразумевает pickup, pickup — approach. Настройки opening≥0.90 и displacement≤0.01 m присутствуют, но в текущем коде это legacy diagnostics, а не обязательные approach gates. Нельзя описывать старую open-hand gate как действующий критерий.

WX200 staged reward устроен иначе: reaching имеет 2 milestones, pickup — 3, placement — 5; terminal return — их достигнутая доля. Он не включает автоматически CDPR three-stage advantage routing.

## 10. Псевдокод

Готовые LaTeX snippets:

- algorithms/staged_bootstrapping.tex — фактическое расписание и conditional SIL текущего WX200 driver;
- algorithms/cdpr_group_update.tex — математическое ядро выбранного CDPR full-task continuation.

Они не подключены к main.tex автоматически: это два разных уровня описания, которые нельзя слить в якобы единый уже выполненный эксперимент. Для включения нужны пакеты algorithm и algpseudocode. Статья и training code в этом аудите не изменены.

Для исторической CDPR цепочки отдельные запуски, warm starts и ручные решения описаны в campaign reports. Найденные threshold values WX200 не подтверждают, что именно по ним когда-то переключалась CDPR lineage.

## 11. Проверка описания

Три существующих CPU-теста прошли: выбор policy по stage metric, deep merge и composed validation metric. Дополнительно исполнены неизменённые методы EarlyStopRule в изоляции от PyTorch: consecutive target, min_steps, plateau и fixed-distance validation gate. Проверены SIL tolerances и seven-stage YAML. Оба LaTeX snippets скомпилированы вместе. PyTorch в локальной среде отсутствует; GPU rollout и численные gradient-тесты не запускались.
