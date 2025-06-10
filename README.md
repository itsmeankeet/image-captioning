Even when you save the complete model architecture using `model.save()`, you still need the custom class definitions in your loading script because:

1. The saved model file contains the architecture configuration and weights, but not the actual Python class implementations. When you save a model, Keras serializes:
   - The model architecture (as a config dictionary)
   - The weights
   - The training config (optional)
   - The optimizer state (optional)

2. During loading, Keras needs to:
   - Parse the architecture config
   - Recreate the layer objects using their class definitions
   - Load the weights into these recreated layers

3. Without the class definitions, Keras doesn't know:
   - How to instantiate your custom layers
   - What the forward pass (`call` method) should do
   - How to handle custom attributes and methods

This is why you need to either:
- Have the custom class definitions in your loading script
- Or provide them via the `custom_objects` parameter when loading

It's similar to how you need class definitions to deserialize objects in any programming language - the serialized data contains the state but not the behavior of the objects.
        