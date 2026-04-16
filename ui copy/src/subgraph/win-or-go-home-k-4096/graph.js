// graph.js
console.log('=== graph.js loaded ===');

// === Feature ===
class Feature {
    /**
     * @param {string} id - Unique feature identifier
     * @param {number} weight - Feature weight
     */
    constructor(id, weight = 1.0) {
        this.id = id;
        this.weight = weight;
    }
}

// === Node ===
class Node {
    /**
     * @param {string} id - Unique identifier
     * @param {string} type - Node type: Embedding / Lorsa / Transcoder / Logit
     * @param {Array<Feature>} [features=[]] - List of Feature objects
     * @param {string} [clerp=''] - Node interpretation string
     * @param {number} position - Node position on the chessboard
     * @param {number} layer - Node layer in the model
     * @param {number} [display_layer] - Optional custom display layer index
     */
    constructor(id, type, features = [], clerp = '', position, layer, display_layer = null) {
        this.id = id;
        this.type = type;
        this.features = features;
        this.clerp = clerp;
        this.position = position;
        this.layer = layer;
        this.display_layer = display_layer;
    }

    getCircleCount() {
        return this.features.length;
    }

    // For backward compatibility, keep childrenIds (derived from feature ids)
    get childrenIds() {
        return this.features.map(feature => feature.id);
    }

    // Add a feature to this node
    addFeature(feature) {
        if (feature instanceof Feature) {
            this.features.push(feature);
        } else {
            console.error('Parameter must be a Feature instance');
        }
    }

    // Get feature by id
    getFeatureById(id) {
        return this.features.find(feature => feature.id === id);
    }

    // Get total feature weight
    getTotalWeight() {
        return this.features.reduce((sum, feature) => sum + feature.weight, 0);
    }
}

// === Edge ===
class Edge{
    /**
     * @param {string} sourceId - Source node id
     * @param {string} targetId - Target node id
     * @param {number} weight - Edge weight
     * @param {boolean} isVirtual - Whether this is a virtual edge (default false)
     */
    constructor(sourceId, targetId, weight, isVirtual = false){
        this.sourceId = sourceId;
        this.targetId = targetId;
        this.weight = weight;
        this.isVirtual = isVirtual;
    }
}

// === Graph ===
class Graph {
    constructor(){
        this.nodes = [];
        this.edges = [];
        this.nodeMap = new Map(); // id -> Node for fast lookup
    }

    addNode(node){
        if (!(node instanceof Node)){
            console.error('paremeter of addNode must be a Node instance');
            return;
        }
        this.nodes.push(node);
        this.nodeMap.set(node.id, node);

    }

    addEdge(edge){
        if (!(edge instanceof Edge)){
            console.error('paremeter of addEdge must be a Edge instance');
            return;
        }
        if (!this.nodeMap.has(edge.sourceId) || !this.nodeMap.has(edge.targetId)){
            console.error(`source (${edge.sourceId}) or target (${edge.targetId}) node not found`);
            return;
        }
        this.edges.push(edge);
    }

    getNodeById(id){
        return this.nodeMap.get(id);
    }

    getChildren(nodeId) {
        const node = this.getNodeById(nodeId);
        if (!node){
            console.error(`node (${nodeId}) not found`);
            return [];
        }
        return node.childrenIds.map(cid => this.getNodeById(cid).filter(n>=n));
    }

    getIncomingEdges(nodeId) {
        return this.edges.filter(e => e.targetId === nodeId);
    }

    getOutgoingEdges(nodeId) {
        return this.edges.filter(e => e.sourceId === nodeId);
    }

    getNodeTypeCounts() {
        const counts = {};
        for (const node of this.nodes) {
            counts[node.type] = (counts[node.type] || 0) + 1;
        }
        return counts;
    }
}

// initialize graph
const graph = new Graph();

// Control switch for custom display layer
const USE_CUSTOM_DISPLAY_LAYER = true;  // Set to true to use custom display_layer, false to use auto-sorted layers

const embeddingNodes = [
    { id: 'E@48', type: 'embedding', position: 48, layer: 0 },
    { id: 'E@52', type: 'embedding', position: 52, layer: 0 },
    { id: 'E@2', type: 'embedding', position: 2, layer: 0 },
    { id: 'E@12', type: 'embedding', position: 12, layer: 0 },
    { id: 'E@10', type: 'embedding', position: 10, layer: 0 },
    { id: 'E@34', type: 'embedding', position: 34, layer: 0 },
]

embeddingNodes.forEach(n => {
    graph.addNode(new Node(n.id, n.type, [], '', n.position, n.layer));
});

const node_M0_2 = new Node(
    'M0#0@2',
    'transcoder',
    [
        new Feature('1_3482_2', 0.8),
        new Feature('1_7372_2', 1.2),
        new Feature('1_11009_2', 0.6)
    ],
    'My K',
    2,    // position
    2,    // layer
    1     // custom display_layer
);

const node_M0_12 = new Node(
    'M0#0@12',
    'transcoder',
    [
        new Feature('1_2175_12', 0.9),
        new Feature('1_6486_12', 1.1),
        new Feature('1_7558_12', 0.7),
        new Feature('1_7812_12', 1.3),
        new Feature('1_11437_12', 0.5)
    ],
    'Opponent\'s Q',
    12,    // position
    2,     // layer
    1      // custom display_layer
);

const node_M0_48 = new Node(
    'M0#0@48',                  // Node id
    'transcoder',               // Node type
    [
        new Feature('1_3026_48', 1.0),
        new Feature('1_5170_48', 0.8),
        new Feature('1_8120_48', 1.2),
        new Feature('1_12234_48', 0.9)
    ],
    'My Q',                     // clerp
    48,                         // position
    2,                          // layer
    1                           // custom display_layer
);

const node_M0_52 = new Node(
    'M0#0@52',
    'transcoder',
    [new Feature('1_3662_52', 1.1)],
    'Opponent\'s K',
    52,    // position
    2,     // layer
    1      // custom display_layer
);

const node_A4_52 = new Node(
    'A4#0@52',
    'lorsa',
    [
        new Feature('8_6813_52', 0.9),
        new Feature('8_7125_52', 1.1)
    ],
    'My Q can threaten O\'s K',
    52,    // position
    9,     // layer
    2      // custom display_layer
);

// const node_A4_2 = new Node(
//     'A4#0@2',
//     'lorsa',
//     ['8_9732_2'],
//     'Attend to O\'s pieces',
//     2,    // position
//     9      // layer
// );

// const node_A5_16 = new Node(
//     'A5#0@16',
//     'lorsa',
//     ['10_3219_16'],
//     'Can check here',
//     16,    // position
//     11      // layer
// );

// const node_A5_2 = new Node(
//     'A5#0@2',
//     'lorsa',
//     ['10_9774_2'],
//     'My K attends to both Q',
//     2,    // position
//     11      // layer
// );

const node_A5_34 = new Node(
    'A5#0@34',
    'lorsa',
    [new Feature('10_3374_34', 1.0)],
    'Q\'s destination to diagonally attack O\'s K',
    34,    // position
    11,    // layer
    3      // custom display_layer
);

// const node_A7_52 = new Node(
//     'A7#0@52',
//     'lorsa',
//     ['14_150_52'],
//     'O\'s K, attend to destination of my B/Q that can diagonally attack O\'s K',
//     52,    // position
//     15      // layer
// );
const node_A8_10 = new Node(
    'A8#0@10',
    'lorsa',
    [new Feature('16_3338_10', 1.2)],
    'My K can be threatened by O\'s Q',
    10,    // position
    17,    // layer
    3      // custom display_layer
);

const node_A11_34 = new Node(
    'A11#0@34',
    'lorsa',
    [new Feature('22_3654_34', 1.0)],
    'RECHECK: A diagonal attack by my Q',
    34,    // position
    23,    // layer
    4      // custom display_layer
);

const node_A12_34 = new Node(
    'A12#0@34',
    'lorsa',
    [new Feature('24_10273_34', 1.0)],
    'DEFENSE: Protect my K from O\'s Q',
    34,    // position
    25,    // layer
    5     // custom display_layer
);

// Additional Logit node
const node_L_34 = new Node(
    'L#0@34',
    'logit',
    [],
    "Output: 'a2a4'",
    34,   // position
    30,   // layer (higher than the current maximum layer)
    6     // custom display_layer
);

graph.addNode(node_M0_2);
graph.addNode(node_M0_12);
graph.addNode(node_M0_48);
graph.addNode(node_M0_52);
graph.addNode(node_A4_52);
graph.addNode(node_A5_34);
graph.addNode(node_A8_10);
graph.addNode(node_A11_34);
graph.addNode(node_A12_34);
graph.addNode(node_L_34);
const edges = [
    { source: 'E@2', target: 'M0#0@2', weight: 1.0 },
    { source: 'E@12', target: 'M0#0@12', weight: 1.0 },
    { source: 'E@48', target: 'M0#0@48', weight: 1.0 },
    { source: 'E@52', target: 'M0#0@52', weight: 1.0 },
    { source: 'M0#0@48', target: 'A4#0@52', weight: 1.0 },
    { source: 'M0#0@52', target: 'A4#0@52', weight: 1.0 },
    { source: 'M0#0@2', target: 'A8#0@10', weight: 1.0 },
    { source: 'M0#0@12', target: 'A8#0@10', weight: 1.0 },

    { source: 'A4#0@52', target: 'A5#0@34', weight: 1.0 },
    { source: 'A5#0@34', target: 'A11#0@34', weight: 1.0 },
    { source: 'A8#0@10', target: 'A12#0@34', weight: 1.0 },

    // Edges pointing to the final Logit output node
    { source: 'A11#0@34', target: 'L#0@34', weight: 1.0 },
    { source: 'A12#0@34', target: 'L#0@34', weight: 1.0 },

];

// Virtual edges - indicate positional correspondence
const virtualEdges = [
    { source: 'E@10', target: 'A8#0@10', weight: 1.0, isVirtual: true },
    { source: 'E@34', target: 'A5#0@34', weight: 1.0, isVirtual: true },
];

// Create edges from (source, target, weight)
edges.forEach(e => {
    graph.addEdge(new Edge(e.source, e.target, e.weight));
});

// Add virtual edges
virtualEdges.forEach(e => {
    graph.addEdge(new Edge(e.source, e.target, e.weight, e.isVirtual));
});

// Usage examples: working with features on a node
console.log('=== Feature Usage Examples ===');

// Example 1: inspect all features on a node
const m0_2_node = graph.getNodeById('M0#0@2');
console.log(`Node ${m0_2_node.id} has ${m0_2_node.features.length} features:`);
m0_2_node.features.forEach(feature => {
    console.log(`  - ${feature.id}: weight=${feature.weight}`);
});

// Example 2: compute total feature weight
console.log(`Total weight for ${m0_2_node.id}: ${m0_2_node.getTotalWeight()}`);

// Example 3: find a specific feature by id
const feature = m0_2_node.getFeatureById('1_7372_2');
if (feature) {
    console.log(`Found feature: ${feature.id} with weight ${feature.weight}`);
}

// Example 4: backward-compatibility test for childrenIds
console.log(`childrenIds (backward compatibility): ${m0_2_node.childrenIds.join(', ')}`);

console.log('========================');