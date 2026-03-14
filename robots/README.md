# Robots Folder

Fork users can place new embodiments here.

Suggested structure:

```text
robots/
  my_robot/
    robot.xml
    controller.py
    joint_metadata.json
    collision_limits.yaml
    README.md
```

Then point your config at:

- `embodiment.robot_root`
- `embodiment.xml_path`
- `embodiment.controller.file`
- `embodiment.action_adapter`

If your controller exposes different method names than the CDPR example, map them through `embodiment.controller.method_map`.
