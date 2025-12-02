window.initGridsnap = function({
  gridData = [],
  maxX = null,
  isFullScreenY = true,
  pad = 10,
  gridSizeY = 80,
  isFillContainer,
  sel = d3.select('.gridsnap'),
  repositionFn = null,
  serializedGrid = '',
} = {}){
  var gridsnap = {gridData, serializeGrid, deserializeGrid}

  gridData.forEach((d, i) => {
    d.next = {...d.cur}
    d.class = d.class === undefined ? i : d.class
  })
  
  var maxX = maxX || d3.max(gridData, d => d.cur.x + d.cur.w)
  function calcgridSizeX(){ 
    return (sel.node().offsetWidth)/maxX
  }
  var gridSizeX = calcgridSizeX()
  
  // TODO: bubble events first
  function calcGridSizeY() {
    if (!isFillContainer) return gridSizeY
    return sel.node().offsetHeight/(d3.max(gridData, d => d.cur.y + d.cur.h) || 1)
  }
  
  gridSizeY = calcGridSizeY()
  
  var resizeKey = 'resize.gridsnap' + serializedGrid
  d3.select(window).on(resizeKey, util.throttle(() => {
    var newX = calcgridSizeX()
    var newY = calcGridSizeY()
    if (newX == gridSizeX && newY == gridSizeY) return

    gridSizeX = newX
    gridSizeY = newY
    renderPositions()
    
    
    gridItemSel.each(d => d.resizeFn?.())
  }, 500))

  var gridsnapSel = sel.html('').append('div.gridsnap-container')

  var gridItemSel = gridsnapSel.appendMany('div.grid-item', gridData)
    .each(function(d){ d.sel = d3.select(this).append('div.grid-contents').classed(d.class, 1) })

  gridItemSel.append('div.move-handle')
    .text('✣')
    .call(makeDragFn(false))
  gridItemSel.append('div.resize-handle')
    .text('↘')
    .call(makeDragFn(true))

  var previewSel = gridsnapSel.append('div.preview.grid-item')

  function makeDragFn(isResize) {
    return d3.drag()
      .subject((ev, d) => ({
        x: (d.cur.x + (isResize ? d.cur.w : 0))*gridSizeX,
        y: (d.cur.y + (isResize ? d.cur.h : 0))*gridSizeY
      }))
      .container(function(){ return this.parentNode.parentNode })
      .on('start', function(ev, d){
        gridData.forEach(d => d.next = {...d.cur})
        d.dragStart = {...d.cur}

        gridsnapSel.classed('dragging', 1)
        d3.select(this.parentNode).classed('dragging', 1)
        previewSel.st({'display': ''})
      })
      .on('end', (ev, d) => {
        gridData.forEach(d => d.cur = {...d.next})

        gridsnapSel.classed('dragging', 0)
        gridItemSel.classed('dragging', 0)
        previewSel.st({'display': 'none'})
        renderPositions()

        if (isResize) d.resizeFn?.()
      })
      .on('drag', isResize ? resize : drag)

    function drag(ev, d) {
      d.cur.x = ev.x/gridSizeX
      d.cur.y = ev.y/gridSizeY

      pushGrid(d)
      renderPositions(d)
    }

    function resize(ev, d) {
      d.cur.x = d.dragStart.x
      d.cur.y = d.dragStart.y

      d.cur.w = ev.x/gridSizeX - d.dragStart.x
      d.cur.h = ev.y/gridSizeY - d.dragStart.y
      if (d.cur.w < 0) {
        d.cur.x += d.cur.w
        d.cur.w = d.dragStart.w - d.cur.w
      }
      if (d.cur.h < 0) {
        d.cur.y += d.cur.h
        d.cur.h = d.dragStart.h - d.cur.h
      }

      pushGrid(d)
      renderPositions(d)
    }
  }


  function pushGrid(active) {
    if (active) active.next = snapToGrid(active.cur)

    var sortedGridData = d3.sort(gridData, d => d != active)
    sortedGridData = d3.sort(sortedGridData, d => d == active ? d.cur.y : d.next.y)

    var topArray = d3.range(maxX).map(d => 0)
    sortedGridData.forEach(d => {
      var {x, y, w, h} = d.next
      d.next.y = d3.max(d3.range(w), i => topArray[x + i])
      d3.range(w).forEach(i => topArray[x + i] = d.next.y + h)
    })

    function snapToGrid({x, y, w, h}) {
      var rv = {x: Math.max(0, Math.round(x)), y: Math.max(0, Math.round(y)), w: Math.max(1, Math.round(w)), h: Math.max(1, Math.round(h))}
      if (rv.x + rv.w > maxX) rv.x = Math.max(0, maxX - rv.w)
      return rv
    }
  }

  function renderPositions(active){
    gridItemSel.call(renderGridItem, 'next')

    if (active){
      gridItemSel.filter(d => d == active).call(renderGridItem, 'cur')
      previewSel.datum(active).call(renderGridItem, 'next')
    } else{
      if (!isFillContainer){
        var maxY = Math.max(maxY, d3.max(gridData, d => d.next.y + d.next.h))
        gridsnapSel.st({height: Math.max(isFullScreenY ? window.innerHeight : 0, maxY*gridSizeY + pad) + 'px'})
      }
    }

    repositionFn?.(gridsnap)

    function renderGridItem(itemSel, key) {
      itemSel
        .translate(d => [d[key].x*gridSizeX + pad/2, d[key].y*gridSizeY + pad/2].map(Math.round))
        .st({
          width: d => Math.round(Math.max(0, d[key].w*gridSizeX - pad)), // negative sizes bug out
          height: d => Math.round(Math.max(0, d[key].h*gridSizeY - pad))
        })
    }
  }

  function serializeGrid(){
    return gridData.map(d => {
      var {x, y, w, h} = d.cur
      return `${d.class}${[x, y, w, h].map(d3.format('02d')).join('')}`
    }).join('_')
  }

  function deserializeGrid(serializedGrid){
    serializedGrid?.split('_').forEach(str => {
      var match = str.match(/^([\w-]+)(\d{8})$/)
      if (!match) return
      var [_, className, coords] = match

      var [x, y, w, h] = d3.range(4).map(i => +coords.substr(i*2, 2))
      var gridItem = gridData.find(d => d.class == className)
      if (gridItem) gridItem.next = {x, y, w, h}
    })

    pushGrid()
    gridData.forEach(d => d.cur = {...d.next})
    renderPositions()
  }

  deserializeGrid(serializedGrid)

  return gridsnap
}

window.init?.()
