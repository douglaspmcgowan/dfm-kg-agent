# Building a DFM Knowledge Graph from Machining Forum Data
### A Step-by-Step Tutorial: Reddit JSON → LlamaIndex Property Graph → Neo4j

---

## What You're Building

This tutorial walks you through turning raw Reddit posts from machining communities (r/CNC, r/CAM, r/Machinists) into a queryable **Design for Manufacturing (DFM) Knowledge Graph**. The end result is a structured database where you can ask questions like:

- "What design features most often cause chatter when machining stainless steel?"
- "What are the tribal rules for deep-pocket milling that aren't in textbooks?"
- "Which failure modes are machinists most afraid of with thin-walled aluminum parts?"

The pipeline runs in four stages:

1. **Ingest** — Load and clean Reddit JSON
2. **Define** — Tell the AI exactly what DFM concepts to look for (the ontology)
3. **Extract** — Run LlamaIndex's schema-constrained extractor to build graph paths
4. **Store & Query** — Push to Neo4j and run research queries

---

## Why This Approach

Standard LLM summarization gives you text back. That's useful for reading, but not for research. A **Property Graph** gives you a structure you can traverse, query, and analyze statistically — "how many posts link thin walls to chatter?" requires a graph, not a chat interface.

LlamaIndex's `PropertyGraphIndex` with a `SchemaLLMPathExtractor` is the right tool here because:

- It enforces a strict DFM ontology, so the AI can't invent nodes like "that guy at the shop" or "yesterday's problem"
- It preserves metadata (upvotes, subreddit, date) as properties on nodes and edges — critical for validating tribal knowledge
- Its retrieval layer is built for RAG from the ground up, so querying later is straightforward

---

## Prerequisites

Before you write a line of code, you need three things running:

**Python environment** (3.10+ recommended):
```bash
pip install llama-index llama-index-graph-stores-neo4j llama-index-llms-openai
```

**OpenAI API key** — the extractor uses GPT-4o to do the actual entity/relation pulling. Set it as an environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

**Neo4j instance** — the easiest option for getting started is [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/), which has a free tier. Create an instance, save your connection URL, username, and password — you'll need them below.

---

## Step 1: Load and Clean Your Reddit JSON

Reddit data is messy. Posts contain tangential comments, jokes, mod announcements, and noise that will pollute your graph. Before extraction, filter aggressively.

Your Reddit JSON should be a list of post objects. The relevant fields are `selftext` (the post body), `ups` (upvotes), and `subreddit`. If you're pulling from PRAW or a local scrape, the structure will look roughly like this:

```json
[
  {
    "selftext": "We had a job last week — 6061 thin walls, 0.060 inch. Chatter was brutal until we slowed the feed to 40% and bumped RPM. Never doing that feature without a roughing pass again.",
    "ups": 312,
    "subreddit": "CNC",
    "created_utc": 1705000000
  }
]
```

Here's the cleaning and loading script:

```python
import json
from llama_index.core import Document

# Keywords that suggest a post contains tribal/experiential knowledge
SIGNAL_KEYWORDS = [
    "hard to machine", "pro tip", "lesson learned", "learned the hard way",
    "chatter", "deflection", "thin wall", "deep pocket", "tolerance",
    "fixturing", "surface finish", "tool wear", "runout", "setup time",
    "never do", "always", "rule of thumb", "from experience"
]

def has_signal(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in SIGNAL_KEYWORDS)

def load_reddit_posts(filepath: str) -> list[Document]:
    with open(filepath) as f:
        posts = json.load(f)
    
    documents = []
    for post in posts:
        text = post.get("selftext", "").strip()
        
        # Skip deleted/removed posts and posts without substance
        if not text or text in ("[deleted]", "[removed]"):
            continue
        
        # Skip posts shorter than 100 characters — usually just links or one-liners
        if len(text) < 100:
            continue
        
        # Optionally: only keep posts with domain-relevant content
        # Remove this filter if you want everything and plan to clean later
        if not has_signal(text):
            continue
        
        doc = Document(
            text=text,
            metadata={
                "upvotes": post.get("ups", 0),
                "subreddit": post.get("subreddit", "unknown"),
                "created_utc": post.get("created_utc", 0),
                "source": "reddit"
            }
        )
        documents.append(doc)
    
    print(f"Loaded {len(documents)} documents after filtering")
    return documents
```

A note on the signal filter: this is intentionally conservative. If you're unsure, skip it and let the schema extractor in Step 3 handle the noise — it will simply produce no paths for posts with nothing relevant.

---

## Step 2: Define Your DFM Ontology

This is the most important design decision in the entire pipeline. The ontology tells the AI what "counts" as a node and what "counts" as a relationship. If you're too broad, the graph fills with noise. If you're too narrow, you miss valuable content.

For DFM research targeting machining tribal knowledge, the following schema captures the concepts that machinists actually talk about:

**Nodes (the "nouns" of machining):**

| Node Type | What It Represents | Examples |
|---|---|---|
| `Feature` | A geometric characteristic of the part being designed | thin wall, deep pocket, internal corner, blind hole, undercut |
| `Material` | The workpiece material | AL6061, 304 stainless, Inconel 718, hardened steel |
| `Tooling` | Cutting tools or tool configurations | end mill, ball nose, insert, stub length, long reach |
| `Failure_Mode` | Something that goes wrong | chatter, deflection, poor surface finish, tool breakage, work hardening |
| `Heuristic` | A rule of thumb or tribal practice | a machinist's personal policy or hard-won lesson |
| `Constraint` | An operational or machine limitation | tolerance spec, available RPM range, fixturing limitation |

**Relationships (the "verbs" of DFM):**

| Relationship | Meaning |
|---|---|
| `CAUSES` | X leads to Y under some condition |
| `REQUIRES` | X needs Y to work correctly |
| `PREVENTS` | X stops Y from happening |
| `DIFFICULT_IN` | X is hard to achieve with Y |
| `RECOMMENDS` | X suggests using or doing Y |
| `CONTRADICTS` | X is a heuristic that disagrees with Y |

Here's the schema definition in code:

```python
from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

# Node types — Literal enforces exact string matching
entities = Literal[
    "Feature",
    "Material",
    "Tooling",
    "Failure_Mode",
    "Heuristic",
    "Constraint"
]

# Relationship types
relations = Literal[
    "CAUSES",
    "REQUIRES",
    "PREVENTS",
    "DIFFICULT_IN",
    "RECOMMENDS",
    "CONTRADICTS"
]

# Validation schema — defines which nodes can connect via which relationships
# This prevents nonsensical edges like "Heuristic DIFFICULT_IN Constraint"
validation_schema = {
    "Feature": ["CAUSES", "REQUIRES", "DIFFICULT_IN"],
    "Material": ["CAUSES", "DIFFICULT_IN", "REQUIRES"],
    "Tooling": ["PREVENTS", "RECOMMENDS", "REQUIRES"],
    "Failure_Mode": ["CAUSES", "REQUIRES"],
    "Heuristic": ["PREVENTS", "CAUSES", "RECOMMENDS", "CONTRADICTS"],
    "Constraint": ["REQUIRES", "CAUSES"]
}

kg_extractor = SchemaLLMPathExtractor(
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True  # Forces the LLM to use only your defined schema — no freeform nodes
)
```

The `strict=True` flag is essential. Without it, the LLM will invent nodes for incidental details mentioned in posts — things like "my buddy" or "last Tuesday's job" — which pollute the graph and make queries unreliable.

---

## Step 3: Build the Property Graph

With documents loaded and the extractor configured, you can now build the graph. This step makes repeated calls to OpenAI (one per document chunk), so expect it to take a few minutes and cost a few dollars on a large dataset.

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex

# Connect to Neo4j
# Replace these with your Aura or local instance credentials
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="your-aura-password",
    url="bolt+s://your-instance-id.databases.neo4j.io:7687"
)

# Load your documents (from Step 1)
documents = load_reddit_posts("reddit_machining.json")

# Build the index
# This is the main extraction step — it calls GPT-4o for each document chunk
index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    kg_extractors=[kg_extractor],
    show_progress=True  # Prints a progress bar — useful for large datasets
)

# Save the index locally so you don't have to rebuild it
index.storage_context.persist(persist_dir="./dfm_kg_storage")

print("Graph built and saved.")
```

What's happening under the hood: LlamaIndex chunks each document, sends each chunk to GPT-4o with instructions derived from your schema, and parses the response into (subject, relation, object) triplets. These triplets are then written to Neo4j as nodes and directed edges, with your document metadata attached as properties.

**Loading a saved index later** (so you don't rebuild from scratch every time):

```python
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./dfm_kg_storage")
index = load_index_from_storage(storage_context)
```

---

## Step 4: Query the Graph

Once built, the graph can be queried in two ways: natural language (for exploration and prototyping) and Cypher (for precise research queries).

### Method A: Natural Language Query Engine

The easiest way to interact with the graph. Under the hood, LlamaIndex retrieves relevant subgraphs and passes them to the LLM for synthesis.

```python
query_engine = index.as_query_engine(include_text=True)

# Example research queries
queries = [
    "What design features most commonly cause chatter in machined parts?",
    "What are the tribal rules machinists use for thin-walled aluminum parts?",
    "What tooling do experienced machinists recommend for deep pockets in stainless?",
    "Which heuristics contradict standard textbook recommendations?"
]

for q in queries:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print(f"A: {response}")
```

### Method B: Direct Cypher Queries (Research Mode)

For your research paper, this is where the real power is. Cypher lets you ask structural questions about the graph — not just "what does the graph say about X" but "how many posts connect X to Y, and how strong is the signal?"

Open Neo4j Browser (available in the Aura dashboard) and run these queries directly.

**Find all failure modes linked to thin-wall features:**
```cypher
MATCH (f:Feature {name: "thin wall"})-[:CAUSES]->(failure:Failure_Mode)
RETURN f.name, failure.name, count(*) as frequency
ORDER BY frequency DESC
```

**Find the most cited heuristics in the dataset:**
```cypher
MATCH (h:Heuristic)-[r]->(n)
RETURN h.name, type(r), n.name, count(*) as citations
ORDER BY citations DESC
LIMIT 20
```

**Find heuristics that contradict each other — potential research findings:**
```cypher
MATCH (h1:Heuristic)-[:CONTRADICTS]->(h2:Heuristic)
RETURN h1.name, h2.name
```

**Find which materials are considered most difficult:**
```cypher
MATCH (m:Material)-[:DIFFICULT_IN]->(constraint)
RETURN m.name, count(*) as difficulty_mentions
ORDER BY difficulty_mentions DESC
```

**Find the full neighborhood around a specific failure mode:**
```cypher
MATCH (n)-[r]-(failure:Failure_Mode {name: "chatter"})
RETURN n, r, failure
LIMIT 50
```

The last query produces a visual graph in Neo4j Browser — nodes connected by labeled edges, colored by type. This is directly usable as a figure in your paper.

---

## Step 5: Entity Resolution (Deduplication)

Before running analysis, you need to handle the fact that machinists use multiple names for the same thing. "6061", "AL6061", "6061-T6", and "aluminum 6061" are all the same material node in your DFM domain — but the extractor will create four separate nodes.

There are two approaches, from simple to thorough:

**Option A: Rule-based normalization (fast, good enough for a first pass)**

Create a normalization map and run a Cypher merge before querying:

```python
NORMALIZATIONS = {
    # Materials
    "6061": "AL6061",
    "6061-t6": "AL6061",
    "aluminum 6061": "AL6061",
    "304": "304 Stainless",
    "304ss": "304 Stainless",
    "stainless": "304 Stainless",  # may be too aggressive — use with care
    # Features
    "thin wall": "Thin_Wall",
    "thin-wall": "Thin_Wall",
    "thin walled": "Thin_Wall",
    "deep pocket": "Deep_Pocket",
    "deep pockets": "Deep_Pocket",
}
```

```cypher
// Example: merge all "6061" nodes into "AL6061"
MATCH (a:Material {name: "6061"}), (b:Material {name: "AL6061"})
CALL apoc.refactor.mergeNodes([a, b], {properties: "combine"})
YIELD node
RETURN node
```

Note: the Cypher merge requires the APOC library, which is pre-installed on Neo4j Aura.

**Option B: LLM-assisted deduplication (more thorough)**

For each pair of same-type nodes with similar names, ask an LLM whether they refer to the same concept. Only merge on a confident "yes":

```python
from openai import OpenAI

client = OpenAI()

def should_merge(name_a: str, name_b: str, node_type: str) -> bool:
    prompt = f"""
    In the context of CNC machining and DFM (design for manufacturing):
    Do "{name_a}" and "{name_b}" (both type: {node_type}) refer to the same concept?
    Answer only YES or NO.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().upper() == "YES"
```

Run this over all same-type node pairs, then execute the Cypher merge on confirmed duplicates. This is slower and costs more API calls, but produces a cleaner graph.

---

## Step 6: Connecting to Your Frame Extraction Pipeline

If you're running a separate frame extraction pipeline (using Pydantic schemas and structured outputs), you can feed the extracted frames into the graph rather than raw post text. This gives you cleaner, pre-validated input to the KG extractor.

The key is treating each `KnowledgeFrame` as a document:

```python
import json
from llama_index.core import Document

def frames_to_documents(extractions_path: str) -> list[Document]:
    with open(extractions_path) as f:
        extractions = json.load(f)
    
    documents = []
    for post_extraction in extractions:
        for frame in post_extraction.get("frames", []):
            # Use the source quote as the document text — it's already validated
            text = frame.get("source_quote", "")
            if not text:
                continue
            
            doc = Document(
                text=text,
                metadata={
                    "frame_type": frame.get("frame_type"),
                    "main_point": frame.get("main_point"),
                    "author": frame.get("author_username"),
                    "source_url": frame.get("source_url"),
                    "post_date": frame.get("post_date"),
                    "post_reaction_count": frame.get("post_reaction_count", 0),
                    "directness": frame.get("directness"),
                    "expression_strength": frame.get("expression_strength"),
                }
            )
            documents.append(doc)
    
    return documents
```

This approach has several advantages: the source quotes are shorter and more focused than full posts, the frame type metadata gives the KG extractor a hint about what kind of knowledge to look for, and the reaction counts and credibility data flow through as edge properties — enabling weighted graph analysis.

---

## What the Graph Lets You Do (Research Value)

The structured graph enables research questions that are impossible to answer from raw text or embeddings alone:

**Frequency analysis** — "Which failure modes appear most often across the corpus?" This is a count query, not a semantic search.

**Co-occurrence analysis** — "Which materials and features co-occur with the highest-severity failure modes?" This requires traversing multiple hops in the graph.

**Contradiction detection** — "Are there heuristics in the graph that directly contradict each other?" These are your most interesting research findings — contradictions often reflect shop-context dependencies that the graph can help surface.

**Comparison with textbooks** — Build a second small graph from DFM handbooks and compare node/edge distributions. Nodes that appear frequently in forum data but not in handbooks are your "tribal knowledge gaps."

**Credibility weighting** — Because author credibility tier and post reaction count flow through as metadata, you can weight your analysis toward high-credibility sources and see whether high-credibility advice differs from the broader corpus.

---

## Putting It All Together

Here's the complete pipeline as a single runnable script:

```python
import json
import os
from typing import Literal

from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI

# --- Config ---
REDDIT_JSON_PATH = "reddit_machining.json"
NEO4J_URL = os.environ["NEO4J_URL"]
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
STORAGE_DIR = "./dfm_kg_storage"

# --- Step 1: Load documents ---
def load_documents(path):
    with open(path) as f:
        posts = json.load(f)
    docs = []
    for post in posts:
        text = post.get("selftext", "").strip()
        if not text or text in ("[deleted]", "[removed]") or len(text) < 100:
            continue
        docs.append(Document(
            text=text,
            metadata={"upvotes": post.get("ups", 0), "subreddit": post.get("subreddit", "")}
        ))
    return docs

# --- Step 2: Define schema ---
entities = Literal["Feature", "Material", "Tooling", "Failure_Mode", "Heuristic", "Constraint"]
relations = Literal["CAUSES", "REQUIRES", "PREVENTS", "DIFFICULT_IN", "RECOMMENDS", "CONTRADICTS"]
validation_schema = {
    "Feature": ["CAUSES", "REQUIRES", "DIFFICULT_IN"],
    "Material": ["CAUSES", "DIFFICULT_IN", "REQUIRES"],
    "Tooling": ["PREVENTS", "RECOMMENDS", "REQUIRES"],
    "Failure_Mode": ["CAUSES", "REQUIRES"],
    "Heuristic": ["PREVENTS", "CAUSES", "RECOMMENDS", "CONTRADICTS"],
    "Constraint": ["REQUIRES", "CAUSES"]
}
kg_extractor = SchemaLLMPathExtractor(
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True
)

# --- Step 3: Connect to Neo4j and build graph ---
graph_store = Neo4jPropertyGraphStore(
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    url=NEO4J_URL
)

documents = load_documents(REDDIT_JSON_PATH)
print(f"Building graph from {len(documents)} documents...")

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    kg_extractors=[kg_extractor],
    show_progress=True
)

index.storage_context.persist(persist_dir=STORAGE_DIR)
print(f"Done. Graph stored in Neo4j and saved locally to {STORAGE_DIR}")

# --- Step 4: Run a sample query ---
query_engine = index.as_query_engine(include_text=True)
result = query_engine.query("What are the most common causes of chatter in CNC machining?")
print("\nSample query result:")
print(result)
```

Run it:
```bash
NEO4J_URL="bolt+s://your-id.databases.neo4j.io:7687" \
NEO4J_PASSWORD="your-password" \
python build_dfm_kg.py
```

---

## Troubleshooting

**"No paths extracted from document"** — The post likely doesn't contain content that matches your schema. This is expected behavior for noise posts. Check your filtering in Step 1 if you're getting this for most documents.

**"Connection refused" from Neo4j** — Check that your URL starts with `bolt+s://` for Aura (TLS required) or `bolt://` for local instances.

**Rate limit errors from OpenAI** — Add a sleep between document batches, or reduce batch size. The `show_progress=True` flag helps you see where you are.

**Duplicate nodes after extraction** — Run entity resolution (Step 5) before doing frequency analysis. Raw extraction will always produce duplicates.

**Graph is too sparse** — Loosen the signal keyword filter in Step 1, or remove it entirely and let the schema extractor filter implicitly.

---

## Next Steps

Once the graph is built and validated, three natural extensions follow:

**Human validation** — Sample 100–200 extracted triplets and verify them manually. This gives you a precision metric to report and catches systematic extraction errors.

**Comparison corpus** — Scrape a DFM handbook or textbook, build a second smaller graph with the same schema, and compare. The delta between "what forums talk about" and "what textbooks cover" is a core research finding.

**RAG system** — Wire the graph to a retrieval system that can take a part description (or eventually a CAD screenshot) and return relevant heuristics, risks, and recommendations. This is the DFM feedback proof of concept that the research is building toward.
