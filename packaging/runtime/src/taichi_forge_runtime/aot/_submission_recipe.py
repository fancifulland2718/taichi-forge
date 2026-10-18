"""Cold, pointer-free submission choices; replay retains only native commands."""


def validate_submission_recipe(recipe, operations, binding_strategy=None):
    if recipe is None:
        return
    if binding_strategy is not None:
        raise ValueError("Vulkan submission recipes cannot use CUDA binding frames")
    if not isinstance(recipe, dict) or type(recipe.get("version")) is not int or recipe["version"] != 1:
        raise ValueError("Unsupported submission recipe version")
    kind = recipe.get("kind")
    if kind == "vulkan_segmented":
        if set(recipe) != {"kind", "version", "ends"}:
            raise ValueError("Segmented Vulkan submission requires exact operation ends")
        ends = recipe["ends"]
        if not isinstance(ends, list) or not ends:
            raise ValueError("Segmented Vulkan submission needs nonempty ends")
        previous = 0
        for end in ends:
            if type(end) is not int or not previous < end <= len(operations):
                raise ValueError("Vulkan submission ends must increase within the replay sequence")
            previous = end
        if previous != len(operations):
            raise ValueError("Vulkan submission ends must cover the replay sequence")
    elif kind == "vulkan_independent_graphics":
        if set(recipe) != {"kind", "version"} or len(operations) < 3:
            raise ValueError("Independent Vulkan submission needs prefix, graphics and consumer")
    else:
        raise ValueError("Unsupported submission recipe kind")
    if any(isinstance(op, dict) and op.get("kind") in ("branch", "while", "cuda_parallel") for op in operations):
        raise ValueError("Vulkan submission recipes require a flat replay sequence")
