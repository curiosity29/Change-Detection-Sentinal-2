

# def Inner(model, layer_name):
#   return tf.keras.Model(inputs = model.input, outputs = model.get_layer("down2").output)
# inner = Inner(model, "down2")

# prediction = inner.predict(np.array([image_]))[0]

# for channel_ in range(prediction.shape[-1]):
#   plt.imshow(prediction[..., channel_], cmap = "gray")
#   plt.show()


# ##

# att = inner = Inner(model, "multi_head_attention_6")
# prediction = att.predict(np.array([images_data_train[73]]))[0]
# prediction.shape


# for channel_ in range(10):
#   plt.imshow(prediction[..., channel_], cmap = "gray")
#   plt.show()