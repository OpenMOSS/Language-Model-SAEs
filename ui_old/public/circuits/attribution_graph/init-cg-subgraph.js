window.initCgSubgraph = function ({visState, renderAll, data, cgSel}) {
  var subgraphSel = cgSel.select('.subgraph')
  subgraphSel.datum().resizeFn = renderSubgraph

  var nodeIdToNode = {}
  var sgNodes = []
  var sgLinks = []

  let nodeSel = null
  let memberNodeSel = null
  let simulation = null

  var nodeWidth = 75
  var nodeHeight = 25

  function supernodesToUrl() {
    // util.params.set('supernodes', JSON.stringify(subgraphState.supernodes))
  }

  var subgraphState = visState.subgraph = visState.subgraph || {
    sticky: true,
    dagrefy: true,
    supernodes: visState.supernodes || [],
    activeGrouping: {
      isActive: false,
      selectedNodeIds: new Set(),
    }
  }
  
  d3.select('body')
    .on('keydown.grouping' + data.metadata.slug, ev => {
      if (ev.repeat) return
      if (!visState.isEditMode || ev.key != 'g') return
      subgraphState.activeGrouping.isActive = true
      styleNodes()
      
      subgraphSel.classed('is-grouping', true)
    })
    .on('keyup.grouping' + data.metadata.slug, ev => {
      if (!visState.isEditMode || ev.key != 'g') return
      if (subgraphState.activeGrouping.selectedNodeIds.size > 1){
        var allSelectedIds = []
        var prevSupernodeLabel = ''
        subgraphState.activeGrouping.selectedNodeIds.forEach(id => {
          var node = nodeIdToNode[id]
          if (!node?.memberNodeIds) return allSelectedIds.push(id)
          prevSupernodeLabel = node.ppClerp

          // if a supernode is selected, remove the previous super node
          subgraphState.supernodes = subgraphState.supernodes.filter(([label, ...nodeIds]) =>
            !nodeIds.every(d => node.memberNodeIds.includes(d))
          )
          // and adds its member nodes to selection
          node.memberNodeIds.forEach(id => allSelectedIds.push(id))
        })

        var label = prevSupernodeLabel || allSelectedIds
          .map(id => nodeIdToNode[id]?.ppClerp)
          .find(d => d) || 'supernode'
        subgraphState.supernodes.push([label, ...new Set(allSelectedIds)])
        supernodesToUrl()
      }
      subgraphState.activeGrouping.isActive = false
      subgraphState.activeGrouping.selectedNodeIds.clear()
      renderSubgraph()
      
      subgraphSel.classed('is-grouping', false)
    })

  let {nodes, links} = data
  
  function renderSubgraph() {
    var c = d3.conventions({
      sel: subgraphSel.html(''),
      margin: {top: 26, bottom: 5, left: visState.isHideLayer ? 0 : 30},
      layers: 'sd',
    })
    // subgraphSel.st({borderTop: '1px solid #eee'})
    
    c.svg.append('text.section-title').text('Subgraph').translate(-16, 1)
    c.svg.append('g.border-path').append('path')
      .at({stroke: '#eee', d: 'M 0 -10 H ' + c.width})


    var [svg, div] = c.layers

    // // set up arrowheads
    // svg.appendMany('marker', [{id: 'mid-negative', color: '#40004b'},{id: 'mid-positive', color: '#00441b'}])
    //   .at({id: d => d.id, orient: 'auto', refX: .1, refY: 1}) // marker-height/marker-width?
    //   .append('path')
    //   .at({d: 'M0,0 V2 L1,1 Z', fill: d => d.color})


    // pick out the subgraph and do supernode surgery
    nodes.forEach(d => d.supernodeId = null)
    var pinnedIds = visState.pinnedIds.slice(0, 200) // max of 200 nodes
    var pinnedNodes = nodes.filter(d => pinnedIds.includes(d.nodeId))

    // create supernodes and mark their children
    nodeIdToNode = Object.fromEntries(pinnedNodes.map(d => [d.nodeId, d]))
    var supernodes = subgraphState.supernodes
      .map(([label, ...nodeIds], i) => {
        var nodeId = nodeIdToNode[label] ? `supernode-${i}` : label
        var memberNodes = nodeIds.map(id => nodeIdToNode[id]).filter(d => d)
        memberNodes.forEach(d => d.supernodeId = nodeId)
  
        var rv = {
          nodeId,
          featureId: `supernode-${i}`,
          ppClerp: label,
          layer: d3.mean(memberNodes, d => +d.layer),
          ctx_idx: d3.mean(memberNodes, d => d.ctx_idx),
          ppLayer: d3.extent(memberNodes, d => +d.layer).join('—'),
          streamIdx: d3.mean(memberNodes, d => d.streamIdx),
          memberNodeIds: nodeIds,
          memberNodes,
          isSuperNode: true,
        }
        nodeIdToNode[rv.nodeId] = rv
  
        return rv
      })
      .filter(d => d.memberNodes.length)
    
    // update clerps — fragile hack if hClerpUpdate changes
    nodes.forEach(d => d.ppClerp = d.clerp)
    supernodes.forEach(({ppClerp, memberNodes}) => {
      if (memberNodes.length == 1 && ppClerp == memberNodes[0].ppClerp) return
      
      memberNodes.forEach(d => {
        d.ppClerp = `[${ppClerp}]` + (ppClerp != d.ppClerp ? ' ' + d.ppClerp : '')
      })
    })
    
    // inputAbsSumExternalSn: the abs sum of input links from outside the supernode
    pinnedNodes.forEach(d => {
      d.inputAbsSumExternalSn = d3.sum(d.sourceLinks, e => {
        if (!e.sourceNode.supernodeId) return Math.abs(e.weight)
        return e.sourceNode.supernodeId == d.supernodeId ? 0 : Math.abs(e.weight)
      })
      d.sgSnInputWeighting = d.inputAbsSumExternalSn/d.inputAbsSum
    })

    // subgraph plots pinnedNodes not in a supernode and supernodes
    sgNodes = pinnedNodes.filter(d => !d.supernodeId).concat(supernodes)
    sgNodes.forEach(d => {
      // for supernodes, sum up values from member nodes
      if (d.isSuperNode) {
        d.inputAbsSum = d3.sum(d.memberNodes, e => e.inputAbsSum)
        d.inputAbsSumExternalSn = d3.sum(d.memberNodes, e => e.inputAbsSumExternalSn)
      } else {
        d.memberNodes = [d]
      }

      var sum = d3.sum(d.memberNodes, e => e.sgSnInputWeighting)
      d.memberNodes.forEach(e => e.sgSnInputWeighting = e.sgSnInputWeighting/sum)
    })

    // select subgraph links
    sgLinks = links
      .filter(d => nodeIdToNode[d.sourceNode.nodeId] && nodeIdToNode[d.targetNode.nodeId])
      .map(d => ({
        source: d.sourceNode.nodeId,
        target: d.targetNode.nodeId,
        weight: d.weight,
        color: d.pctInputColor,
        ogLink: d,
      }))

    // then remap source/target to supernodes
    sgLinks.forEach(link => {
      if (nodeIdToNode[link.source]?.supernodeId) link.source = nodeIdToNode[link.source].supernodeId
      if (nodeIdToNode[link.target]?.supernodeId) link.target = nodeIdToNode[link.target].supernodeId
    })

    // finally combine parallel links and remove self-links
    sgLinks = d3.nestBy(sgLinks, d => d.source + '-' + d.target)
      .map(links => {
        var weight = d3.sum(links, link => {
          var {inputAbsSumExternalSn, sgSnInputWeighting} = link.ogLink.targetNode
          return link.weight/inputAbsSumExternalSn*sgSnInputWeighting
        })

        return {
          source: links[0].source,
          target: links[0].target,
          weight,
          color: utilCg.pctInputColorFn(weight),
          pctInput: weight,
          pctInputColor: utilCg.pctInputColorFn(weight),
          ogLinks: links
        }
      })
      .filter(d => d.source !== d.target)
    sgLinks = d3.sort(sgLinks, d => Math.abs(d.weight))

    let xScale = d3.scaleLinear()
      .domain(d3.extent(sgNodes.map(d => d.ctx_idx)))
      .range([0, c.width*3/4])
    let yScale = d3.scaleLinear()
      .domain(d3.extent(sgNodes.map(d => d.streamIdx)).toReversed())
      .range([0, c.height - nodeHeight])

    // d3.force is impure, need copy
    // Also want to persist these positions across node changes
    const existingNodes = window.selForceNodes && new Map(window.selForceNodes.map(n => [n.node.nodeId, n]))
    window.selForceNodes = sgNodes.map(node => {
      const existing = existingNodes?.get(node.nodeId)
      return {
        x: existing ? existing.x : xScale(node.ctx_idx),
        y: existing ? existing.y : yScale(node.streamIdx),
        fx: existing?.fx,
        fy: existing?.fy,
        nodeId: node.nodeId, // for addFeatureEvents
        featureId: node.featureId, // for addFeatureEvents
        node,
        sortedSlug: d3.sort(node.memberNodes.map(d => d.featureIndex).join(' ')),
      }
    })
    

    var selForceNodes = window.selForceNodes = d3.sort(window.selForceNodes, d => d.sortedSlug)
    window._exportSubgraphPos = function(){
      return selForceNodes.map(d => [d.x/c.width*1000, d.y/c.height*1000].map(Math.round)).flat().join(' ')
    }

    if (simulation) simulation.stop()
    simulation = d3.forceSimulation(selForceNodes)
      .force('link', d3.forceLink(sgLinks).id(d => d.node.nodeId))
      .force('charge', d3.forceManyBody())
      .force('collide', d3.forceCollide(Math.sqrt(nodeHeight ** 2 + nodeWidth ** 2) / 2))
      .force('container', forceContainer([[-10, 0], [c.width - nodeHeight, c.height - nodeHeight]]))
      .force('x', d3.forceX(d => xScale(d.node.ctx_idx)).strength(.1))
      .force('y', d3.forceY(d => yScale(d.node.streamIdx)).strength(2))

    var svgPaths = svg.appendMany('path.link-path', sgLinks).at({
      fill: 'none',
      markerMid: d => d.weight > 0 ? 'url(#mid-positive)' : 'url(#mid-negative)',
      strokeWidth: d => Math.abs(d.weight)*15,
      stroke: d => d.color,
      opacity: 0.8,
      strokeLinecap: 'round',
    })

    var edgeLabels = svg.appendMany('text.weight-label', sgLinks)
      // .text(d => d3.format('+.2f')(d.weight))

    simulation.on('tick', renderForce)

    var drag = d3.drag()
      .on('drag', (ev) => {
        // Only when actually dragging, mark as no longer dagre positioned and restart sim
        ev.subject.dagrePositioned = false
        if (!ev.active) simulation.alphaTarget(0.3).restart()
        ev.subject.fx = ev.subject.x = ev.x
        ev.subject.fy = ev.subject.y = ev.y
        renderForce()
      })
      .on('end', (ev) => {
        if (!ev.active) simulation.alphaTarget(0)
        if (!subgraphState.sticky && !ev.subject.dagrePositioned){
          ev.subject.fx = null
          ev.subject.fy = null
        }
      })

    nodeSel = div
      .appendMany('div.supernode-container', selForceNodes)
      .translate(d => [d.x, d.y])
      .st({width: nodeWidth, height: nodeHeight})
      .call(utilCg.addFeatureEvents(visState, renderAll, ev => ev.shiftKey))
      .on('click.group', (ev, d) => {
        var {isActive, selectedNodeIds} = subgraphState.activeGrouping
        if (!isActive) return

        // If it's a child node, use its parent supernode's ID instead
        var nodeId = d.supernodeId || d.nodeId
        selectedNodeIds.has(nodeId) ? selectedNodeIds.delete(nodeId) : selectedNodeIds.add(nodeId)
        
        styleNodes()
        ev.stopPropagation()
        ev.preventDefault()
      })
      .call(drag)

    selForceNodes.forEach(d => {
      if (!d.node.memberNodes) d.node.memberNodes = [d.node]
    })

    var supernodeSel = nodeSel//.filter(d => d.node.isSuperNode)
      .classed('is-supernode', true)
      .st({height: nodeHeight + 12})

    memberNodeSel = supernodeSel.append('div.member-circles')
      .st({
        width: d => d.node.memberNodes.length <= 4 ? 'auto' : 'calc(32px + 12px)', 
        gap: d => d.node.memberNodes.length <= 4 ? 4 : 0,
      })
      .appendMany('div.member-circle', d => d.node.memberNodes)
      .classed('not-clt-feature', d => d.feature_type != 'cross layer transcoder')
      .st({marginLeft: function(d, i) {
          var n = this.parentNode.childNodes.length  
          return n <= 4 ? 0 : i == 0 ? 0 : -((n - 4)*8)/(n - 1)
      }})
      .call(utilCg.addFeatureEvents(visState, renderAll, ev => ev.shiftKey))
      .on('click.stop-parent', ev => {
        if (!subgraphState.activeGrouping.isActive) ev.stopPropagation()
      })  
      .on('mouseover.stop-parent', ev => ev.stopPropagation())
      .at({title: d => d.ppClerp})

    if (visState.isEditMode) {
      // TODO: enable
      supernodeSel.select('.member-circles')
        .filter(d => d.node.isSuperNode)
        .append('div.ungroup-btn')
        .text('×').st({top: 2, left: -15, position: 'absolute'})
        .on('click', (ev, d) => {
          ev.stopPropagation()
          
          subgraphState.supernodes = subgraphState.supernodes.filter(([label, ...nodeIds]) =>
            !nodeIds.every(id => d.node.memberNodeIds.includes(id))
          )
          supernodesToUrl()
          renderSubgraph()
        })
    }

    var nodeTextSel = nodeSel.append('div.node-text-container')
    nodeTextSel.append('span')
      .text(d => d.node.ppClerp)
      .on('click', (ev, d) => {
        if (!visState.isEditMode) return
        if (!d.node.isSuperNode) return
        // TODO: enable?
        return
        ev.stopPropagation()

        var spanSel = d3.select(ev.target).st({display: 'none'})
        var input = d3.select(spanSel.node().parentNode).append('input')
          .at({class: 'temp-edit', value: spanSel.text()})
          .on('blur', save)
          .on('keydown', ev => {
            if (ev.key === 'Enter'){
              save()
              input.node().blur()
            }
            ev.stopPropagation()
          })

        input.node().focus()

        function save(){
          var idx = subgraphState.supernodes.findIndex(([label, ...nodeIds]) =>
            nodeIds.every(id => d.node.memberNodeIds.includes(id))
          )
          if (idx >= 0){
            subgraphState.supernodes[idx][0] = input.node().value || 'supernode'
            supernodesToUrl()
            renderSubgraph()
          }
        }
      })


    nodeTextSel.each(function(d) {
      d.textHeight = this.getBoundingClientRect().height || -8
    })

    nodeSel.append('div.clicked-weight.source')
    nodeSel.append('div.clicked-weight.target')
    styleNodes()


    var checkboxes = Object.entries({
      sticky: () => {
        // simulation.alphaTarget(0.3).restart()
        if (!subgraphState.sticky) unsticky()
      },
      dagrefy: () => {
        subgraphState.dagrefy ? dagrefy() : selForceNodes.forEach(d => d.dagrePositioned = null)
      },
    }).map(([key, fn]) => ({key, fn}))


    if (visState.isEditMode && false) {
      div.append('div.checkbox-container').translate([-c.margin.left, c.margin.bottom])
        .appendMany('label', checkboxes).append('input')
        .at({type: 'checkbox'})
        .property('checked', d => subgraphState[d.key])
        .on('change', function(ev, d){
          subgraphState[d.key] = this.checked
          d.fn()
        })
        .parent().append('span').text(d => d.key)
    }

    checkboxes.forEach(d => d.fn())

    function unsticky(){
      selForceNodes.forEach(d => (d.fx = d.fy = null))
      simulation.alphaTarget(0.3).restart()
      if (subgraphState.dagrefy) {
        subgraphState.dagrefy = false
        d3.select('.checkbox-container').selectAll('input').filter(d => d.key == 'dagrefy').property('checked', 0)
        checkboxes.find(d => d.key == 'dagrefy').fn()
      }
    }

    function dagrefy(){
      if (visState.sg_pos){
        var nums = visState.sg_pos.split(' ').map(d => +d)
        selForceNodes.forEach((d, i) => {
          d.fx = d.x = nums[i*2 + 0]/1000*c.width
          d.fy = d.y = nums[i*2 + 1]/1000*c.height
        })

        nodeSel.translate(d => [d.x, d.y])
        styleNodes()
        renderEdges()

        visState.og_sg_pos = visState.sg_pos
        delete visState.sg_pos
      }
      if (visState.og_sg_pos) return


      var g = new window.dagre.graphlib.Graph()
      g.setGraph({rankdir: 'BT', nodesep: 20, ranksep: 20})
      g.setDefaultEdgeLabel(() => ({}))

      sgLinks.forEach(d =>{
        if (Math.abs(d.weight) < .003) return
        // set width and height to make dagre return x and y for edges
        g.setEdge(d.source.nodeId, d.target.nodeId, {width: 1, height: 1, labelpos: 'c', weight: Math.abs(d.weight)})
      })
      sgNodes.forEach(d => {
        g.setNode(d.nodeId, {width: nodeWidth, height: nodeHeight})
      })

      window.dagre.layout(g)

      // rescale to fit container
      var xs = d3.scaleLinear([0, g.graph().width], [0, Math.min(c.width, g.graph().width)])
      var ys = d3.scaleLinear([0, g.graph().height], [0, Math.min(c.height, g.graph().height)])

      // flip to make ctx_idx left to right and streamIdx bottom to top
      var w0 = d3.mean(selForceNodes, d =>  g.node(d.nodeId).x*d.node.ctx_idx)
      var w1 = d3.mean(selForceNodes, d => -g.node(d.nodeId).x*d.node.ctx_idx)
      if (w0 < w1) xs.range(xs.range().reverse())

      var w0 = d3.mean(selForceNodes, d => g.node(d.nodeId).y*d.node.streamIdx)
      var w1 = d3.mean(selForceNodes, d => -g.node(d.nodeId).y*d.node.streamIdx)
      if (w0 < w1) ys.range(ys.range().reverse())

      for (var node of window.selForceNodes) {
        var pos = g.node(node.nodeId)
        node.fx = node.x = xs(pos.x) - nodeWidth/2
        node.fy = node.y = ys(pos.y) - nodeHeight/2
        node.dagrePositioned = true
      }

      // var curveFactory = d3.line(d => d.x, d => d.y).curve(d3.curveBasis)
      // svgPaths.at({d: d => {
      //   var points = g.edge(d.source.nodeId, d.target.nodeId)?.points
      //   if (!points) return ''
      //   return curveFactory(points.map(p => ({x: xs(p.x), y: ys(p.y)})))
      // }})
      renderEdges()

      // edgeLabels.translate(d => {
      //   var pos = g.edge(d.source.nodeId, d.target.nodeId)
      //   if (!pos) return [-100, -100]
      //   return [xs(pos.x), ys(pos.y)]
      // })
      styleNodes()
    }

    function renderForce(){
      nodeSel.translate(d => [d.x, d.y])

      renderEdges()

      edgeLabels
        .filter(d => !(d.source.dagrePositioned && d.target.dagrePositioned))
        .translate(d => [
          (d.source.x + d.target.x) / 2 + nodeWidth / 2,
          (d.source.y + d.target.y) / 2 + nodeHeight / 2
        ])
    }
    
    function renderEdges(){
      
      // TODO: use actual strokeWidth to spread
      d3.nestBy(sgLinks, d => d.source).forEach(links => {
        // if (links[0].source.nodeId == '6_12890134_-0') debugger
        var numSlots = links[0].source.node.memberNodes.length
        var totalWidth = (Math.min(4, numSlots))*8
        d3.sort(links, d => Math.atan2(d.target.y - d.source.y, d.target.x - d.source.x))
          .forEach((d, i) => d.sourceOffsetX = (i - links.length/2)*totalWidth/links.length)
      })

      d3.nestBy(sgLinks, d => d.target).forEach(links => {
        var numSlots = links[0].target.node.memberNodes.length
        var totalWidth = (Math.min(4, numSlots) + 1)*3
        d3.sort(links, d => -Math.atan2(d.source.y - d.target.y, d.source.x - d.target.x))
          .forEach((d, i) => d.targetOffsetX = (i - links.length/2)*totalWidth/links.length)
      })

      svgPaths.at({
        d: d => {
          var x0 = d.source.x + nodeWidth/2 + d.sourceOffsetX
          var y0 = d.source.y 
          var x1 = d.target.x + nodeWidth/2 + d.targetOffsetX
          var y1 = d.target.y + d.target.textHeight + 28

          return `M${x0},${y0} L${x1},${y1}`
        },
      })
    }
  }


  function styleNodes() {
    if (!nodeSel) return

    nodeSel
      .classed('clicked', d => d.nodeId == visState.clickedId)
      .classed('hovered', d => d.featureId == visState.hoveredId)
      .st({zIndex: d => Math.round(d.x*20 + d.y) + 1000})
      .classed('grouping-selected', d => subgraphState.activeGrouping.selectedNodeIds.has(d.nodeId))

    memberNodeSel
      .classed('clicked', d => d.nodeId == visState.clickedId)
      .classed('hovered', d => d.featureId == visState.hoveredId)
      .st({
        background: d => d.tmpClickedLink?.pctInputColor,
        color: d => utilCg.bgColorToTextColor(d.tmpClickedLink?.pctInputColor)
      })
      // .at({title: d => d3.format('.1%')(d.tmpClickedLink?.pctInput)})



    // style clicked links using supernode adjusted graph when possible
    sgNodes.forEach(d => {
      d.tmpClickedSgSource = d.tmpClickedLink?.sourceNode == d ? d.tmpClickedLink : null
      d.tmpClickedSgTarget = d.tmpClickedLink?.targetNode == d ? d.tmpClickedLink : null
    })

    if (visState.clickedId) {
      sgLinks.forEach(d => {
        if (d.source.nodeId == visState.clickedId) nodeIdToNode[d.target.nodeId].tmpClickedSgTarget = d
        if (d.target.nodeId == visState.clickedId) nodeIdToNode[d.source.nodeId].tmpClickedSgSource = d
      })
    }

    // nodeSel.selectAll('.clicked-weight.source')
    //   .st({display: d => d.node.tmpClickedSgSource ? '' : 'none'})
    //   .filter(d => d.node.tmpClickedSgSource)
    //   .text(d => d3.format('.1%')(d.node.tmpClickedSgSource.pctInput))
    //   .st({
    //     background: d => d.node.tmpClickedSgSource.pctInputColor,
    //     color: d => utilCg.bgColorToTextColor(d.node.tmpClickedSgSource.pctInputColor)
    //   })

    // nodeSel.selectAll('.clicked-weight.target')
    //   .st({display: d => d.node.tmpClickedSgTarget ? '' : 'none'})
    //   .filter(d => d.node.tmpClickedSgTarget)
    //   .text(d => d3.format('.1%')(d.node.tmpClickedSgTarget.pctInput))
    //   .st({
    //     background: d => d.node.tmpClickedSgTarget.pctInputColor,
    //     color: d => utilCg.bgColorToTextColor(d.node.tmpClickedSgTarget.pctInputColor)
    //   })
  }

  renderAll.hClerpUpdate.fns['subgraph'] = renderSubgraph
  renderAll.pinnedIds.fns['subgraph'] = renderSubgraph
  renderAll.clickedId.fns['subgraph'] = styleNodes
  renderAll.hoveredId.fns['subgraph'] = styleNodes

  // https://github.com/1wheel/d3-force-container/blob/master/src/force-container.js
  function forceContainer(bbox) {
    var nodes, strength = 1

    function force(alpha) {
      var i,
          n = nodes.length,
          node,
          x = 0,
          y = 0

      for (i = 0; i < n; ++i) {
        node = nodes[i], x = node.x, y = node.y

        if (x < bbox[0][0]) node.vx += (bbox[0][0] - x)*alpha
        if (y < bbox[0][1]) node.vy += (bbox[0][1] - y)*alpha
        if (x > bbox[1][0]) node.vx += (bbox[1][0] - x)*alpha
        if (y > bbox[1][1]) node.vy += (bbox[1][1] - y)*alpha
      }
    }

    force.initialize = function(_){
      nodes = _
    }

    force.bbox = function(_){
      return arguments.length ? (bbox = +_, force) : bbox
    }
    force.strength = function(_){
      return arguments.length ? (strength = +_, force) : strength
    }

    return force
  }
}

window.init?.()
