window.initCgNodeConnections = function({visState, renderAll, data, cgSel}){
  
  var nodeConnectionsSel = cgSel.select('.node-connections')
  var headerSel = null
  var clickedNode = null
  var featureSel = null

  function render() {
    nodeConnectionsSel.html('').st({display: 'flex', flexDirection: 'column'})
    clickedNode = data.nodes.find(e => e.nodeId === visState.clickedId)    
    headerSel = nodeConnectionsSel.append('div.header-top-row.section-title').st({marginBottom: 20}).datum(clickedNode)
    
    if (!clickedNode) return headerSel.text('Click a feature on the left for details')
    
    addHeaderRow(headerSel) 

    var types = [
      { id: 'input', title: 'Input Features'},
      { id: 'output', title: 'Output Features' }
    ]
    types.forEach(type => type.sections = ['Positive', 'Negative'].map(title => {
      var nodes = data.nodes.filter(d => {
        var weight = type.id === 'input' ? d.tmpClickedSourceLink?.weight : d.tmpClickedTargetLink?.weight
        return title == 'Positive' ? weight > 0 : weight < 0
      })
      nodes = d3.sort(nodes, d => -(type.id === 'input' ? d.tmpClickedSourceLink?.pctInput : d.tmpClickedTargetLink?.pctInput))
      return {title, nodes}
    }))

    var typesSel = nodeConnectionsSel.append('div.connections')
      .st({flex: '1 0 auto', display: 'flex', overflow: 'hidden', gap: '20px'})
      .appendMany('div.features', types)
      .classed('output', d => d.id === 'output')
      .classed('input', d => d.id === 'input')

    typesSel.append('div.section-title').text(d => d.title)
      .st({marginBottom: 0})
    // addInputSumTable(typesSel.filter(d => d.id == 'input').append('div.sum-table').append('div.section'))

    var featuresContainerSel = typesSel.append('div.effects')

    var sectionSel = featuresContainerSel.appendMany('div.section', d => d.sections)
    featureSel = sectionSel.appendMany('div.feature-row', d => d.nodes)
      .classed('clicked', e => e.featureId == visState.clickedId)

    classPinned()
    classHidden()

    typesSel.each(function(type) {
      d3.select(this).selectAll('.feature-row')
        .call(utilCg.renderFeatureRow, visState, renderAll, type.id === 'input' ? 'tmpClickedSourceLink' : 'tmpClickedTargetLink')
    })
  }
  
  function addHeaderRow(headerSel){
    if (!clickedNode) return 
    
    headerSel.append('text')
      .text(clickedNode.feature_type == 'cross layer transcoder' ? 'F#' + d3.format('08')(clickedNode.feature) : ' ')
      .st({display: 'inline-block', marginRight: 5, 'font-variant-numeric': 'tabular-nums', width: 82})
    headerSel.append('span.feature-icon').text(utilCg.featureTypeToText(clickedNode.feature_type))
    headerSel.append('span.feature-title').text(clickedNode.ppClerp)
    
    // add cmd-click toggle to title
    headerSel.on('click', ev => {
      utilCg.clickFeature(visState, renderAll, clickedNode, ev.metaKey || ev.ctrlKey)

      if (visState.clickedId) return
      // double render to toggle on hoveredId, could expose more of utilCg.clickFeature to prevent
      visState.hoveredId = clickedNode.featureId
      renderAll.hoveredId()
    })
  }

  function addInputSumTable(sumSel){
    return // TODO: turn back on? 
    
    var clickedNode = data.nodes.idToNode[visState.clickedId]
    if (!clickedNode) return
    var inputSum = d3.sum(clickedNode.sourceLinks, d => Math.abs(d.weight))

    var tableSel = sumSel.append('table')
    tableSel.appendMany('th', ['% Feat', '% Err', '% Emb']).text(d => d)

    var trSel = tableSel.appendMany('tr.data', [
      {str: '←', links: clickedNode.sourceLinks.filter(d => d.sourceNode.ctx_idx != clickedNode.ctx_idx)},
      {str: '↓', links: clickedNode.sourceLinks.filter(d => d.sourceNode.ctx_idx == clickedNode.ctx_idx)}
    ])

    trSel.append('td').text(d => d.str).at({title: d => d.str == '↓' ? 'cur token' : 'prev token'})
    trSel.appendMany('td', d => {
        var rv = [
          d.links.filter(e => !e.sourceNode.isError && e.sourceNode.feature_type != 'embedding'),
          d.links.filter(e => e.sourceNode.isError),
          d.links.filter(e => e.sourceNode.feature_type == 'embedding'),
        ]

        if (rv.flat().length != d.links.length) console.error("Non-feature/error/embedding node present")
        return rv
      })
      .html(links => {
        var pos = d3.sum(links.filter(d => d.weight > 0), d => d.weight)/inputSum
        var neg = d3.sum(links.filter(d => d.weight < 0), d => d.weight)/inputSum

        var outStr = `+${d3.format('.2f')(pos)}&#160;&#160;&#160;−${d3.format('.2f')(neg)}`
        return outStr.replaceAll('0.', '.').replaceAll('−−', '−')
      })
  }

  renderAll.hClerpUpdate.fns.push(render)
  renderAll.hoveredId.fns['nodeconnections'] = () => featureSel?.classed('hovered', e => e.featureId == visState.hoveredId)
  renderAll.pinnedIds.fns['nodeconnections'] = classPinned
  renderAll.hiddenIds.fns['nodeconnections'] = classHidden
  renderAll.clickedId.fns['nodeconnections'] = render
  

  function classPinned(){
    var pinnedIdSet = new Set(visState.pinnedIds)
    featureSel?.classed('pinned', d => pinnedIdSet.has(d.nodeId))
    headerSel?.classed('pinned', d => d && d.nodeId && visState.pinnedIds.includes(d.nodeId))
      .select('span.feature-title').text(d => d.ppClerp)
  }

  function classHidden(){
    var hiddenIdSet = new Set(visState.hiddenIds)
    featureSel?.classed('hidden', d => hiddenIdSet.has(d.featureId))
  }
}

window.init?.()
