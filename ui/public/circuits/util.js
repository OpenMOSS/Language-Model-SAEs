window.util = (function () {
  var params = (function(){
    var rv = {}

    rv.get = key => {
      var url = new URL(window.location)
      var searchParams = new URLSearchParams(url.search)

      var str = searchParams.get(key)
      return str && decodeURIComponent(str)
    }

    rv.getAll = () => {
      var url = new URL(window.location)
      var searchParams = new URLSearchParams(url.search)

      var values = {}
      for (const [key, value] of searchParams.entries()) {
        values[key] = decodeURIComponent(value)
      }
      return values
    }

    rv.set = (key, value) => {
      var url = new URL(window.location)
      var searchParams = new URLSearchParams(url.search)

      if (value === null) {
        searchParams.delete(key)
      } else {
        searchParams.set(key, encodeURIComponent(value))
      }

      url.search = searchParams.toString()
      history.replaceState(null, '', url)
    }

    return rv
  })()
  
  async function getFile(path) {
    // Cache storage 
    var __datacache = window.__datacache = window.__datacache || {}

    // If path starts with /, treat as relative to static_js directory
    if (path.startsWith('/')) {
      path = 'https://transformer-circuits.pub/2025/attribution-graphs' + path
    }

    // Return cached result if available 
    if (!__datacache[path]) __datacache[path] = __fetch()
    return __datacache[path]

    async function __fetch() {
      var res = await fetch(path, {cache: 'force-cache'})
      if (res.status == 500) {
        var resText = await res.text()
        console.log(resText, res) 
        throw '500 error'
      }

      var type = path.replaceAll('..', '').split('.').at(-1)
      if (type == 'csv') {
        return d3.csvParse(await res.text())
      } else if (type == 'npy') {
        return npyjs.parse(await res.arrayBuffer())
      } else if (type == 'json') {
        return await res.json()
      } else if (type == 'jsonl') {
        var text = await res.text()
        return text.split(/\r?\n/).filter(d => d).map(line => JSON.parse(line))
      } else {
        return await res.text()
      }
    }
  }
  
  
  function addAxisLabel(c, xText, yText, title='', xOffset=0, yOffset=0, titleOffset=0){
    c.svg.select('.x').append('g')
      .translate([c.width/2, xOffset + 25])
      .append('text.axis-label')
      .text(xText)
      .at({textAnchor: 'middle', fill: '#000'})

    c.svg.select('.y')
      .append('g')
      .translate([yOffset -30, c.height/2])
      .append('text.axis-label')
      .text(yText)
      .at({textAnchor: 'middle', fill: '#000', transform: 'rotate(-90)'})

    c.svg
      .append('g.axis').at({fontFamily: 'sans-serif'})
      .translate([c.width/2, titleOffset -10])
      .append('text.axis-label.axis-title')
      .text(title)
      .at({textAnchor: 'middle', fill: '#000'})
  }

  function ggPlot(c){
    c.svg.append('rect.bg-rect')
      .at({width: c.width, height: c.height, fill: c.isBlack ? '#000' : '#EAECED'}).lower()
    c.svg.selectAll('.domain').remove()

    c.svg.selectAll('.x text').at({y: 4})
    c.svg.selectAll('.x .tick')
      .selectAppend('path').at({d: 'M 0 0 V -' + c.height, stroke: c.isBlack ? '#444' : '#fff', strokeWidth: 1})

    c.svg.selectAll('.y text').at({x: -3})
    c.svg.selectAll('.y .tick')
      .selectAppend('path').at({d: 'M 0 0 H ' + c.width, stroke: c.isBlack? '#444' : '#fff', strokeWidth: 1})

    ggPlotUpdate(c)
  }

  function ggPlotUpdate(c){
    c.svg.selectAll('.tick').selectAll('line').remove()

    c.svg.selectAll('.x text').at({y: 4})
    c.svg.selectAll('.x .tick')
      .selectAppend('path').at({d: 'M 0 0 V -' + c.height, stroke: c.isBlack ? '#444' : '#fff', strokeWidth: 1})

    c.svg.selectAll('.y text').at({x: -3})
    c.svg.selectAll('.y .tick')
      .selectAppend('path').at({d: 'M 0 0 H ' + c.width, stroke: c.isBlack? '#444' : '#fff', strokeWidth: 1})
  }

  function initRenderAll(fnLabels){
    var rv = {}
    fnLabels.forEach(label => {
      rv[label] = (ev) => Object.values(rv[label].fns).forEach(d => d(ev))
      rv[label].fns = []
    })

    return rv
  }
  
  function attachRenderAllHistory(renderAll, skipKeys=['hoverId', 'hoverIdx']) {
    // Add state pushing to each render function
    Object.keys(renderAll).forEach(key => {
      renderAll[key].fns.push(() => {
        if (skipKeys.includes(key)) return
        var simpleVisState = {...visState}
        skipKeys.forEach(key => delete simpleVisState[key])

        var url = new URL(window.location) 
        if (visState[key] == url.searchParams.get(key)) return
        url.searchParams.set(key, simpleVisState[key])
        history.pushState(simpleVisState, '', url)
      })
    })

    // Handle back/forward navigation
    d3.select(window).on('popstate.updateState', ev => {
      if (!ev.state) return
      ev.preventDefault()
      Object.keys(renderAll).forEach(key => {
        if (skipKeys.includes(key)) return 
        if (visState[key] == ev.state[key]) return
        visState[key] = ev.state[key]
        renderAll[key]()
      })
    })
  }
  
  function throttle(fn, delay){
    var lastCall = 0
    return (...args) => {
      if (Date.now() - lastCall < delay) return
      lastCall = Date.now()
      fn(...args)
    }
  }

  function debounce(fn, delay) {
    var timeout
    return (...args) => {
      clearTimeout(timeout)
      timeout = setTimeout(() => fn(...args), delay)
    }
  }

  function throttleDebounce(fn, delay) {
    var lastCall = 0
    var timeoutId

    return function (...args) {
      clearTimeout(timeoutId)
      var remainingDelay = delay - (Date.now() - lastCall)
      if (remainingDelay <= 0) {
        lastCall = Date.now()
        fn.apply(this, args)
      } else {
        timeoutId = setTimeout(() => {
          lastCall = Date.now()
          fn.apply(this, args)
        }, remainingDelay)
      }
    }
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
  }
  
  function cache(fn){
    var cache = {}
    return function(...args){
      var key = JSON.stringify(args)
      if (!(key in cache)) cache[key] = fn.apply(this, args)
      return cache[key]
    }
  }
  var featureExamplesTooltipSel
  var featureExamples 
  var featureQueue = []
  function attachFeatureExamplesTooltip(sel, getFeatureParams, getNearby){
    if (!featureExamplesTooltipSel){
      featureExamplesTooltipSel = d3.select('body')
        .selectAppend('div.tooltip.feature-examples-tooltip.tooltip-hidden')
        .on('mouseover', mousemove)
        .on('mousemove', mousemove) 
        .on('mouseleave', mouseout)
      
      // Add touch event handler to body
      d3.select('body').on('click.feature-tooltip', ev => {
        // Don't trigger if touch is on tooltip or tooltipped element
        if (ev.target.closest('.feature-examples-tooltip') || ev.target.closest('.feature-examples-tooltipped')) return
        mouseout()
      })

      d3.select(window).on('scroll.feature-examples-tooltip', () => {
        if (featureExamplesTooltipSel.isFading || featureExamplesTooltipSel.isFaded) return
        mouseout()
      })

      featureExamplesTooltipSel.append('div.feature-nav')
      featureExamples = window.initFeatureExamples({
        containerSel: featureExamplesTooltipSel.append('div'),
        hideStaleOutputs: true,
      })

      if (window.__feature_tooltip_queue_timer) __feature_tooltip_queue_timer.stop()
      window.__feature_tooltip_queue_timer = d3.timer(() => {
        if (!featureQueue.length) return
        var feature = featureQueue.pop()
        featureExamples.loadFeature(feature.scan, feature.featureIndex)
      }, 250)
    }

    sel
      .on('mousemove.feature-examples-tooltip', mousemove)
      .on('mouseleave.feature-examples-tooltip', mouseout)
      .on('mouseenter.feature-examples-tooltip', function(ev, d){
        setTimeout(mousemove, 0)
        
        // skip moving if we're just bouncing in and out of the current feature
        if (featureExamplesTooltipSel.cur == d && !featureExamplesTooltipSel.classed('tooltip-hidden')) return 
        featureExamplesTooltipSel.cur = d
        
        featureExamplesTooltipSel.node().scrollTop = -200

        featureExamplesTooltipSel.isFaded = false
        featureExamplesTooltipSel.classed('tooltip-hidden', 0)

        // requires either featureIndex or featureIndices
        var {scan, featureIndex, featureIndices} = getFeatureParams(d)
        featureIndices = featureIndices ?? [featureIndex]
        featureIndex = featureIndex ?? featureIndices[0]

        var buttonSel = featureExamplesTooltipSel.select('.feature-nav').html('')
          .appendMany('div.button', featureIndices)
          .text((_, i) => 'Feature ' + (i + 1))
          .classed('active', idx => idx == featureIndex)
          .on('click', (ev, idx) => {
            featureExamples.renderFeature(scan, idx)
            buttonSel.classed('active', idx2 => idx2 == idx)
          })

        featureExamples.renderFeature(scan, featureIndex)

        d3.selectAll('.feature-examples-tooltipped').classed('feature-examples-tooltipped', 0)
        d3.select(this).classed('feature-examples-tooltipped', 1)

        var snBB = this.getBoundingClientRect()
        var ttBB = featureExamplesTooltipSel.node().getBoundingClientRect()
        var left = d3.clamp(20, (ev.clientX-ttBB.width/2), window.innerWidth - ttBB.width - 20)
        var top = snBB.top > innerHeight - snBB.bottom ?
            snBB.top - ttBB.height - 10 :
            snBB.bottom + 10
        featureExamplesTooltipSel.st({left, top, pointerEvents: 'all'})

        getNearby?.(d).forEach(e => featureQueue.push(e))
      })

    function mousemove(){
      if (window.__ttfade) window.__ttfade.stop()
      featureExamplesTooltipSel.isFading = false
      featureExamplesTooltipSel.isFaded = false
    }

    function mouseout(){
      if (featureExamplesTooltipSel.isFading) return

      if (window.__ttfade) window.__ttfade.stop()
      featureExamplesTooltipSel.isFading = true
      window.__ttfade = d3.timeout(() => {
        featureExamplesTooltipSel.classed('tooltip-hidden', 1).st({pointerEvents: 'none'})
        d3.selectAll('.feature-examples-tooltipped').classed('feature-examples-tooltipped', 0)
        featureExamplesTooltipSel.isFading = false
        featureExamplesTooltipSel.isFaded = true
      }, 250)
    }
  }
  
  async function initGraphSelect(sel, cgSlug){
    var {graphs} = await util.getFile('/data/graph-metadata.json')
    
    var selectSel = sel.html('').append('select.graph-prompt-select')
      .on('change', function() {
        cgSlug = this.value 
        // visState.clickedId = undefined
        util.params.set('slug', this.value)
        render()
      })
    
    var cgSel = sel.append('div.cg-container')
  
    selectSel.appendMany('option', graphs)
      .text(d => {
        var scanName = util.nameToPrettyPrint[d.scan] || d.scan
        var prefix = d.title_prefix ? d.title_prefix + ' ' : ''
        return prefix + scanName + ' — ' + d.prompt
      })
      .attr('value', d => d.slug)
      .property('selected', d => d.slug === cgSlug)
  
    function render() {
      initCg(cgSel.html(''), cgSlug, {
        isModal: true,
        // clickedId: visState.clickedId,
        // clickedIdCb: id => util.params.set('clickedId', id)
      })
      
      var m = graphs.find(g => g.slug == cgSlug)
      if (!m) return
      selectSel.at({title: m.prompt})
    }
    render()
  }
  
  function attachCgLinkEvents(sel, cgSlug, figmaSlug){
    sel
      .on('mouseover', () => util.getFile(`/graph_data/${cgSlug}.json`))
      .on('click', (ev) => {
        ev.preventDefault()
        
        if (window.innerWidth < 900 || window.innerHeight < 500) {
          return window.open(`./static_js/attribution_graphs/index.html?slug=${cgSlug}`, '_blank')
        }
  
        d3.select('body').classed('modal-open', true)
        var contentSel = d3.select('modal').classed('is-active', 1)
          .select('.modal-content').html('')
        
        util.initGraphSelect(contentSel, cgSlug)
        
        util.params.set('slug', cgSlug)
        if (figmaSlug) history.replaceState(null, '', '#' + figmaSlug)
      })
  }
  
  // TODO: tidy
  function ppToken(d){
    return d
  }
  
  function ppClerp(d){
    return d
  }
  
  
  var scanSlugToName = {
    'h35': 'jackl-circuits-runs-1-4-sofa-v3_0',
    '18l': 'jackl-circuits-runs-1-1-druid-cp_0',
    'moc': 'jackl-circuits-runs-12-19-valet-m_0'
  }
  
  var nameToPrettyPrint = {
    'jackl-circuits-runs-1-4-sofa-v3_0': 'Haiku',
    'jackl-circuits-runs-1-1-druid-cp_0': '18L',
    'jackl-circuits-runs-12-19-valet-m_0': 'Model Organism',
    'jackl-circuits-runs-1-12-rune-cp3_0': '18L PLTs',
  }

  
  return {
    scanSlugToName,
    nameToPrettyPrint,
    params,
    getFile,
    addAxisLabel,
    ggPlot,
    ggPlotUpdate,
    initRenderAll,
    attachRenderAllHistory,
    throttle,
    debounce,
    throttleDebounce,
    sleep,
    cache,
    initGraphSelect,
    attachCgLinkEvents,
    ppToken,
    ppClerp,
    attachFeatureExamplesTooltip,
  }
})()

window.init?.()
