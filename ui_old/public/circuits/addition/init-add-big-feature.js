window.initBigAddFeature = function({visState, renderAll, allFeatures}){
  var sel = d3.select('.add-big-feature').html('')

  var numSel = sel.append('div.feature-num.section-title')
  var heatmapSel = sel.append('div.heatmap.operand')
  
  sel.append('div.section-title').text('Token Predictions')
  var logitSel = sel.append('div.logit')
  
  sel.append('div.section-title').text('Encoder UMAP')
  utilAdd.drawUmap(sel.append('div.umap'), {allFeatures, visState, renderAll})
  
  sel.append('div.section-title').text('Decoder UMAP')
  utilAdd.drawUmap(sel.append('div.umap'), {allFeatures, visState, renderAll, type: 'dec'})
  
  // sel.append('div.section-title').text('Joint UMAP')
  // utilAdd.drawUmap(sel.append('div.umap'), {allFeatures, visState, renderAll, type: 'joint'})

  sel.append('div.link').st({marginTop: 20})
    .append('a').html('← Circuit Tracing § Addition Case Study')
    .at({href: '../../methods.html#graphs-addition'})

  
  renderAll.clickIdx.fns.push(async () => {
    heatmapSel.html('')

    var idx = visState.clickIdx
    numSel.text(d => `L${allFeatures.idxLookup[idx].layer}#${d3.format('07d')(idx)}`)

    utilAdd.drawHeatmap(heatmapSel, idx, {isBig: true})
    utilAdd.drawLogits(logitSel, idx, {isBig: true})
  })

}

window.init?.()
