// possible improvements:
// - persist to url in diff view
// - support multiple scatter plots
// - color by x
// - facet by x
// - brush to filter
// - scatter plot of links

window.initCgFeatureScatter = function({visState, renderAll, data, cgSel}){
  var nodes = data.nodes//.filter(d => !d.isLogit)

  var numericCols = Object.entries(nodes[0])
    .filter(([k, v]) => typeof v != 'object' && typeof v != 'function' && !utilCg.keysToSkip.has(k) && isFinite(v))
    .map(([k]) => k)

  var xKey = util.params.get('feature_scatter_x') || 'ctx_idx'
  var yKey = util.params.get('feature_scatter_y') || 'target_influence'
  function addSelect(isX){
    var options = isX ? ['Distribution'].concat(numericCols) : numericCols
    selectSel.append('select').st({marginRight: 10})
      .on('change', function(){
        isX ? xKey = options[this.selectedIndex] : yKey = options[this.selectedIndex]
        isX ? util.params.set('feature_scatter_x', xKey) : util.params.set('feature_scatter_y', yKey)
        renderScales()
      })
      .appendMany('option', options)
      .text(d => d)
      .at({value: d => d})
      .filter(d => isX && d == xKey || !isX && d == yKey).at({selected: 'selected'})
  }

  var sel = cgSel.select('.feature-scatter').html('')
  var selectSel = sel.append('div.select-container').st({marginLeft: 35})
  var chartSel = sel.append('div.chart-container')
  addSelect(1)
  addSelect(0)

  function renderScales(){
    var c = d3.conventions({
      sel: chartSel.html('').append('div'),
      margin: {left: 35, bottom: 30, top: 2, right: 6},
    })

    if (xKey == 'Distribution'){
      d3.sort(d3.sort(nodes, d => +d[yKey]), d => d.feature_type)
        .forEach((d, i) => d.Distribution = i/nodes.length)
    }

    c.x.domain(d3.extent(nodes, d => +d[xKey])).nice()
    c.y.domain(d3.extent(nodes, d => +d[yKey])).nice()

    c.yAxis.ticks(3)
    c.xAxis.ticks(5)
    c.drawAxis()
    util.ggPlot(c)
    util.addAxisLabel(c, xKey + ' →', yKey + ' →', '', 0, 5)

    var nodeSel = c.svg.appendMany('text.node', nodes)
      .translate(d => [c.x(d[xKey]) ?? -2, c.y(d[yKey]) ?? c.height + 2])
      .text(d => utilCg.featureTypeToText(d.feature_type))
      .at({
        fontSize: 7,
        stroke: '#000',
        strokeWidth: .2,
        textAnchor: 'middle',
        dominantBaseline: 'central',
        fill: 'rgba(0,0,0,.1)'
      })
      .call(utilCg.addFeatureTooltip)
      .call(utilCg.addFeatureEvents(visState, renderAll))

    // TODO: add hover circle?
    utilCg.updateFeatureStyles(nodeSel)
    renderAll.hoveredId.fns['featureScatter'] = () => utilCg.updateFeatureStyles(nodeSel)
    renderAll.clickedId.fns['featureScatter'] = () => utilCg.updateFeatureStyles(nodeSel)
    renderAll.pinnedIds.fns['featureScatter'] = () => utilCg.updateFeatureStyles(nodeSel)
    renderAll.hiddenIds.fns['featureScatter'] = () => utilCg.updateFeatureStyles(nodeSel)
  }

  // TODO: awkward, maybe gridsnap/widget inits need to be restructured?
  if (!sel.datum().resizeFn) renderScales()
  sel.datum().resizeFn = renderScales
}

window.init?.()
