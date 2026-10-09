<script setup>
import { onMounted, ref, watch } from "vue";
import { autoScale, decodeMap, toImageData } from "../wafer.js";

const props = defineProps({ code: { type: String, required: true }, size: { type: Number, default: 160 } });
const canvas = ref(null);

function draw() {
  if (!canvas.value) return;
  const map = decodeMap(props.code);
  const img = toImageData(map, autoScale(map, props.size));
  canvas.value.width = img.width;
  canvas.value.height = img.height;
  canvas.value.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
}
onMounted(draw);
watch(() => [props.code, props.size], draw);
</script>

<template>
  <canvas ref="canvas" class="wafer" />
</template>
