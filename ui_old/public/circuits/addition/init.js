window.init = async function () {
  var features_enriched = await util.getFile('/data/addition/features_enriched.json')
  
  var allFeatures = window.allFeatures = features_enriched.features
  allFeatures.idxLookup = Object.fromEntries(allFeatures.map(d => [d.idx, d]))
  console.log(allFeatures)
  
  window.visState = window.visState || {
    clickIdx: util.params.get('clickIdx') || 17574692,
    hoverIdx: null
  }
  if (visState.clickIdx == 'undefined') visState.clickIdx = 17574692
  
  var renderAll = util.initRenderAll(['clickIdx', 'hoverIdx'])
  util.attachRenderAllHistory(renderAll, ['hoverIdx'])


  initBigAddFeature({visState, renderAll, allFeatures})
  initAddConnections({visState, renderAll, allFeatures, type: 'inputs'}) 
  initAddConnections({visState, renderAll, allFeatures, type: 'outputs'})

  renderAll.clickIdx()
}


window.init()
