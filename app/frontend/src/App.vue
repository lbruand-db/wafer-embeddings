<script setup>
import { computed, onMounted, ref } from "vue";
import WaferMap from "./components/WaferMap.vue";

const config = ref({ classes: [], endpoint: "", live_embedding: false });
const label = ref("any");
const waferId = ref("");
const k = ref(12);
const crossLot = ref(true);
const live = ref(false);
const result = ref(null);
const loading = ref(false);
const error = ref("");

async function api(path) {
  const r = await fetch(`api/${path}`);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

async function search(id) {
  loading.value = true;
  error.value = "";
  try {
    const p = new URLSearchParams({ id, k: k.value, cross_lot: crossLot.value, live: live.value });
    result.value = await api(`search?${p}`);
  } catch (e) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
}

async function randomWafer() {
  try {
    const q = label.value === "any" ? "" : `?label=${encodeURIComponent(label.value)}`;
    const w = await api(`wafer${q}`);
    await search(w.id);
  } catch (e) {
    error.value = e.message;
  }
}

function byId() {
  if (waferId.value !== "") search(Number(waferId.value));
}

const query = computed(() => result.value?.query);
onMounted(async () => {
  config.value = await api("config").catch(() => config.value);
  await randomWafer();
});
</script>

<template>
  <header>
    <h1>Wafer-map similarity search</h1>
    <p class="sub">WM-811K · per-die ViT + DINO embeddings · Lakebase Search (lakebase_ann, cosine)</p>
  </header>

  <div class="layout">
    <aside>
      <h2>Query</h2>
      <label>Class
        <select v-model="label">
          <option value="any">any</option>
          <option v-for="c in config.classes" :key="c" :value="c">{{ c }}</option>
        </select>
      </label>
      <button class="primary" :disabled="loading" @click="randomWafer">Random wafer</button>

      <label>Wafer id
        <input v-model="waferId" type="number" min="0" placeholder="e.g. 199460" @keyup.enter="byId" />
      </label>
      <button :disabled="loading || waferId === ''" @click="byId">Search by id</button>

      <h2>Options</h2>
      <label>Neighbours: {{ k }}
        <input v-model.number="k" type="range" min="4" max="24" step="4" />
      </label>
      <label class="check"><input v-model="crossLot" type="checkbox" /> Exclude the query's own lot</label>
      <label class="check" :class="{ disabled: !config.live_embedding }">
        <input v-model="live" type="checkbox" :disabled="!config.live_embedding" />
        Re-embed query via endpoint <code>{{ config.endpoint || "n/a" }}</code>
      </label>
      <button v-if="query" :disabled="loading" @click="search(query.id)">Re-run search</button>
    </aside>

    <main>
      <p v-if="error" class="error">{{ error }}</p>
      <p v-if="loading" class="muted">Searching…</p>

      <section v-if="query" class="query">
        <WaferMap :code="query.map_code" :size="200" />
        <div>
          <h2>#{{ query.id }} · {{ query.label || "unlabeled" }}</h2>
          <p>split <b>{{ query.split }}</b> · lot <code>{{ query.lot }}</code> · {{ query.height }}×{{ query.width }}</p>
          <p class="muted">embedding: {{ result.embedding === "live" ? "live from the endpoint" : "stored (batch job)" }}</p>
          <p v-if="query.label">
            <b>{{ result.same_class }}/{{ result.neighbours.length }}</b> neighbours share the class
            <b>{{ query.label }}</b><span v-if="result.cross_lot"> (other lots only)</span>
          </p>
        </div>
      </section>

      <section v-if="result" class="grid">
        <figure v-for="n in result.neighbours" :key="n.id"
                :class="{ match: query.label && n.label === query.label }"
                @click="search(n.id)" title="Search from this wafer">
          <WaferMap :code="n.map_code" :size="120" />
          <figcaption>#{{ n.id }} · {{ n.label || "unlabeled" }}<br />cos {{ n.similarity.toFixed(3) }}</figcaption>
        </figure>
      </section>
    </main>
  </div>
</template>
