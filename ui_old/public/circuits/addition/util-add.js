window.utilAdd = (function(){
  async function drawHeatmap(sel, id, {isBig, isDelay, s}){
    s = s ?? (isBig ? 3 : 1)

    var margin = isBig ? {right: 0, top: 2, bottom: 40} : {top: 0, left: 2, bottom: 2, right: 0}
    var c = d3.conventions({
      sel: sel.html('').classed('operand', 1),
      margin,
      width: s*100,
      height: s*100,
      layers: 'sc',
    })

    // add axis
    c.x.domain([0, 100])
    c.y.domain([0, 100])

    var tickValues = d3.range(0, 110, isBig ? 10 : 20)
    var tickFormat = isBig ? d => d : d => ''
    c.xAxis.tickValues(tickValues).tickFormat(tickFormat).tickPadding(-2)
    c.yAxis.tickValues(tickValues).tickFormat(tickFormat).tickPadding(-2)

    c.drawAxis()
    c.svg.selectAll('.tick').selectAll('line').remove()
    c.svg.selectAll('.x .tick').append('path').at({d: `M 0 0 V ${-c.height}`})
    c.svg.selectAll('.y .tick').append('path').at({d: `M 0 0 H ${c.width}`})

    // load and draw data
    if (isDelay) await util.sleep(32)
    var gridData = await util.getFile(`/data/addition/heatmap/${id}.npy`)

    var maxVal = d3.max(gridData.data)
    maxVal = .15
    var colorScale = d3.scaleSequential(d3.interpolateOranges).domain([0, 1.4*maxVal]).clamp(1)

    for (var i = 0; i < 100*100; i++){
      var v = gridData.data[i]
      if (v == 0) continue

      var row = Math.ceil(100 - i/100 - 1)
      var col = i % 100

      c.layers[1].fillStyle = colorScale(v)
      c.layers[1].fillRect(col*s, row*s, s, s)
    }
  }
  
  async function drawLogits(sel, id, {isBig, isDelay, s}){
    s = s ?? (isBig ? 3 : 1)
    
    var margin = isBig ? {right: 0, top: 0, bottom: 40} : {top: 0, left: 2, bottom: 2, right: 0}
    var c = d3.conventions({
      sel: sel.html('').classed('operand', 1).st({marginTop: -2}),
      margin,
      width: s*100,
      height: s*10,
      layers: 'sc',
    })

    // add axis
    c.x.domain([0, 100])
    c.y.domain([0, 10])

    var tickValues = d3.range(0, 100, isBig ? 10 : 20)
    var tickFormat = isBig ? d => d : d => ''
    c.xAxis.tickValues(tickValues).tickFormat(d => '_' + (d ? d : '00')).tickPadding(-2)
    c.yAxis.tickValues([0, 4, 8]).tickFormat(d => d + '_ _').tickPadding(-2)

    c.drawAxis()
    c.svg.selectAll('.tick').selectAll('line').remove()
    c.svg.selectAll('.x .tick').append('path').at({d: `M 0 5 V ${0}`})
    c.svg.selectAll('.x .tick').select('text').translate(5, 1)
    c.svg.selectAll('.y .tick').append('path').at({d: `M -5 0 H ${0}`})
    c.svg.selectAll('.y .tick').select('text').translate(-5, 0)

    // load and draw data
    if (isDelay) await util.sleep(32)
    var gridData = await util.getFile(`/data/addition/effects/${id}.npy`)
    
    var mean = d3.mean(gridData.data)
    values = gridData.data.map(d => d - mean)
    
    var maxVal = d3.max(values)
    var colorScale = d3.scaleDiverging(d3.interpolatePRGn).domain([maxVal, 0, -maxVal]).clamp(1)

    for (var i = 0; i < 100*10; i++){
      var v = values[i]
      var row = Math.ceil(10 - i/100 - 1)
      var col = i % 100
      

      c.layers[1].fillStyle = colorScale(v)
      c.layers[1].fillRect(col*s, row*s, s, s)
    }
  }


  function drawUmap(sel, {allFeatures, visState, renderAll, type='enc'}){
    var c = d3.conventions({
      sel: sel.html(''),
      width: 300,
      height: 200,
      margin: {left: 0, top: 0, right: 0, bottom: 0},
    })

    var points = allFeatures
      .map(d => ({
        idx: d.idx,
        x: d['umap_' + type][0],
        y: d['umap_' + type][1],
        d,
      }))
      .filter(d => d.d.inputs.length + d.d.outputs.length > 0)

    c.x.domain(d3.extent(points, d => d.x))
    c.y.domain(d3.extent(points, d => d.y))

    var pointSel = c.svg.appendMany('circle', points)
      .translate(d => [c.x(d.x), c.y(d.y)])
      .at({r: 2, fill: '#000', fillOpacity: .2, stroke: '#000'})
      .call(attachFeatureEvents, {visState, renderAll})

    renderAll.hoverIdx.fns.push(() => {
      pointSel.classed('hover', d => d.idx == visState.hoverIdx)
    })

    renderAll.clickIdx.fns.push(() => {
      var clickFeature = allFeatures.idxLookup[visState.clickIdx]
      var idx2strength = {}
      clickFeature.inputs.forEach(d => idx2strength[d.idx] = d.strength)
      clickFeature.outputs.forEach(d => idx2strength[d.idx] = d.strength)

      points.forEach(d => {
        d.isClicked = d.idx == visState.clickIdx
        d.strength = d.isClicked ? 9999 : idx2strength[d.idx] || 0
        d.fill = d.isClicked ? '#000' : d.strength ? utilAdd.color(d.strength) : '#fff'
      })

      pointSel.at({
        fill: d => d.fill,
        r: d => d.strength ? 4 : 1,
        fillOpacity: d => d.idx == visState.clickIdx || d.fill != '#fff' ? 1 : 0,
        stroke: d => {
          if (d.idx == visState.clickIdx) return '#000'
          if (d.fill == '#fff') return 'rgba(0,0,0,0.2)'
          return d3.rgb(d.fill).darker(3)
        }
      }).classed('unselected', d => d.fill == '#fff')
    })
  }
  
  function attachFeatureEvents(sel, {visState, renderAll}){
    sel
      .call(d3.attachTooltip)
      .on('mouseover', (e, d) => {
        d = d.d || d
        
        visState.hoverIdx = d.idx
        renderAll.hoverIdx()
        
        var ttSel = d3.select('.tooltip').html('').st({padding: 20, paddingBottom: 0, paddingTop: 10})
        
        ttSel.append('div.section-title')
          .text(`L${d.layer}#${d3.format('07d')(d.idx)}`)
        utilAdd.drawHeatmap(ttSel.append('div'), d.idx, {isBig: true, s: 2})

        ttSel.append('div.section-title').text('Token Predictions') 
        utilAdd.drawLogits(ttSel.append('div'), d.idx, {isBig: true, s: 2})
      })
      .on('mouseleave', () => {
        visState.hoverIdx = null
        renderAll.hoverIdx()
      })
      .on('click', (e, d) => {
        visState.clickIdx = d.idx
        renderAll.clickIdx()
      })
  }

  return {
    drawHeatmap,
    drawLogits,
    drawUmap,
    color: d3.scaleDiverging(d3.interpolatePRGn).domain([-1, 0, 1]),
    attachFeatureEvents
  }
})()

window.init?.()
