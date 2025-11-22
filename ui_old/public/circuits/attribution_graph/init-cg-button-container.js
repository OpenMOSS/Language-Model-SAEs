window.initCgButtonContainer = function({visState, renderAll, cgSel}){
  var buttonContainer = cgSel.select('.button-container').html('')
    .st({marginBottom: '10px'})

  var linkTypeSel= buttonContainer.append('div.link-type-buttons')
    .appendMany('div', ['input', 'output', 'either', 'both'])
    .text(d => d[0].toUpperCase() + d.slice(1).toLowerCase())
    .on('click', (ev, d) => {
      visState.linkType = d
      renderAll.linkType()
    })

  renderAll.linkType.fns.push(() => {
    linkTypeSel.classed('active', d => d === visState.linkType)
  })
  var showAllSel = buttonContainer.append('div.toggle-buttons')
    .append('div').text('Show all links')
    .on('click', () => {
      visState.isShowAllLinks = visState.isShowAllLinks ? '' : '1'
      renderAll.isShowAllLinks()
    })

  renderAll.isShowAllLinks.fns.push(() => {
    showAllSel.classed('active', visState.isShowAllLinks)
  })

  var clearButtonsSel = buttonContainer.append('div.toggle-buttons')
    .appendMany('div', ['Clear pinned', 'Clear clicked'])
    .text(d => d)
    .on('click', (ev, d) => {
      if (d == 'Clear pinned') {
        visState.pinnedIds = []
        renderAll.pinnedIds()
      } else {
        visState.clickedId = ''
        renderAll.clickedId()
      }
    })

  cgSel.on('keydown.esc-check', ev => {
    if (ev.key == 'Escape') {
      visState.clickedId = ''
      renderAll.clickedId()
    }
  })

  var resetGridSel = buttonContainer.append('div.toggle-buttons')
    .append('div').text('Reset grid')
    .on('click', () => {
      util.params.set('gridsnap', '') // TODO: this won't work with baked in features
      window.location.reload()
    })

  var onSyncValue = visState.isSyncEnabled || '1'
  var syncButtonSel = buttonContainer.append('div.toggle-buttons')
    .append('div').text('Enable sync')
    .on('click', () => {
      visState.isSyncEnabled = visState.isSyncEnabled ? '' : onSyncValue
      renderAll.isSyncEnabled()
    })

  renderAll.isSyncEnabled.fns.push(() => {
    syncButtonSel.classed('active', visState.isSyncEnabled)
  });

}

window.init?.()
