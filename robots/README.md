# Robots Folder

Robot-specific code should live here, not at the repository root.

Suggested structure:

```text
robots/
  my_robot/
    README.md
    my_robot_mujoco/
      robot.xml
      controller.py
    my_robot_tasks/
      reward.py
      env.py
      scenes.py
```

The included example follows:

```text
robots/
  cdpr/
    cdpr_mujoco/
    cdpr_dataset/
```

Then point your config at:

- `repos.dataset_repo`
- `repos.embodiment_repo`
- `embodiment.robot_root`
- `embodiment.xml_path`
- `embodiment.controller.file`
- `embodiment.action_adapter`

If your controller exposes different method names than the CDPR example, map them through `embodiment.controller.method_map`.
