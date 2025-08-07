window.initAddConnections = function({visState, renderAll, allFeatures, type='inputs'}){
  var sel = d3.select(`.add-connections.${type}`).html('')

  var isInputs = type == 'inputs'
  
  var headerSel = sel.append('div.sticky.connection-header')
  var headerLeftSel = headerSel.append('div.section-title.sticky')
    .text(isInputs ? 'Negative Input Features' : 'Positive Output Features')
  var headerRightSel = headerSel.append('div.section-title.sticky')
    .text(isInputs ? 'Positive Input Features' : 'Negative Output Features')
  
  var featureContainerSel = sel.append('div.feature-container')

  renderAll.clickIdx.fns.push(() => {
    var features = allFeatures.idxLookup[visState.clickIdx][type]
      .map(d => ({...d, layer: allFeatures.idxLookup[d.idx].layer}))

    var featureWidth = 100 
    var gap = 20
    
    var availableWidth = (window.innerWidth - 320 - 90)/2 // middle col and gaps
    var nCols = 2 * Math.max(1, Math.floor(availableWidth/(2*(featureWidth + gap)))) // even number directly, min 1 per side
    var nCols = Math.max(4, nCols)
    
    d3.select('.add-main').st({minWidth: d3.select('.add-main').st({minWidth: ''}).node().offsetWidth})
    
    d3.nestBy(features, d => d.strength > 0).forEach(signGroup => {
      var isPos = signGroup[0].strength > 0
      d3.sort(signGroup, d => -Math.abs(d.strength)).forEach((d, i) => {
        d.j = Math.floor(i/(nCols/2))
        d.i = (isPos ^ isInputs ? 0 : nCols/2) + i%(nCols/2)
      })
    })

    sel.st({
      width: featureWidth*nCols + gap*(nCols - 1) + 3,
      height: (d3.max(features, d => d.j) + 1)*(featureWidth + gap)
    })
    
    headerLeftSel.st({
      width: (featureWidth * nCols/2 + gap * (nCols/2 - 1)) + 'px',
      display: 'inline-block'
    })
    headerRightSel.st({
      width: (featureWidth * nCols/2 + gap * (nCols/2 - 1)) + 'px',
      display: 'inline-block',
      marginLeft: gap + 'px'
    })

    var featureSel = featureContainerSel.html('').appendMany('div.feature', features)
      .call(utilAdd.attachFeatureEvents, {visState, renderAll})
      .translate(d => [d.i*(featureWidth + gap), d.j*(featureWidth + gap)])

    featureSel.append('div').each(function(d){ 
      utilAdd.drawHeatmap(d3.select(this), d.idx, {isDelay: d.j > 6}) 
    })

    var featureLabelSel = featureSel.append('div.feature-label')
    featureLabelSel.append('span')
      .text(d => `L${allFeatures.idxLookup[d.idx].layer}#${d3.format('07d')(d.idx)}`)
    featureLabelSel.append('span.strength')
      .st({
        background: d => utilAdd.color(d.strength),
        color: d => Math.abs(d.strength) < 0.6 ? '#000' : '#fff',
      })
      .text(d => d3.format('+.2f')(d.strength))
  })

  renderAll.hoverIdx.fns.push(() => {
    sel.selectAll('.feature').classed('hovered', d => d.idx == visState.hoverIdx)
  })
}



window.init?.()
