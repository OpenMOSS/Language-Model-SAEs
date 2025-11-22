// Text wrapping functionality for D3.js
// Forked from d3-textwrap — BSD-3

(function(){
    var method,
        verify_bounds,
        resolve_bounds,
        resolve_padding,
        pad,
        dimensions,
        wrap,
        textwrap;

    // use tspans method by default for cleaner SVG output
    method = 'tspans';

    // accept multiple input types as boundaries
    verify_bounds = function(bounds) {
        var bounds_object,
            bounds_function;
        bounds_function = typeof bounds === 'function';
        if (typeof bounds === 'object' && ! bounds.nodeType) {
            if (! bounds.height || ! bounds.width) {
                console.error('text wrapping bounds must specify height and width');
                return false;
            } else {
                return true;
            }
        }
        // convert a selection to bounds
        if (
            bounds instanceof d3.selection ||
            bounds.nodeType ||
            bounds_function ||
            bounds_object
        ) {
            return true;
        // use input as bounds directly
        } else {
            console.error('invalid bounds specified for text wrapping');
            return false;
        }
    };

    resolve_bounds = function(bounds) {
        var properties,
            dimensions,
            result,
            i;
        properties = ['height', 'width'];
        if (typeof bounds === 'function') {
            dimensions = bounds();
        } else if (bounds.nodeType) {
            dimensions = bounds.getBoundingClientRect();
        } else if (typeof bounds === 'object') {
            dimensions = bounds;
        }
        result = Object.create(null);
        for (i = 0; i < properties.length; i++) {
            result[properties[i]] = dimensions[properties[i]];
        }
        return result;
    };

    resolve_padding = function(padding) {
        var result;
        if (typeof padding === 'function') {
            result = padding();
        } else if (typeof padding === 'number') {
            result = padding;
        } else if (typeof padding === 'undefined') {
            result = 0;
        }
        if (typeof result !== 'number') {
            console.error('padding could not be converted into a number');
        } else {
            return result;
        }
    };

    pad = function(dimensions, padding) {
        var padded;
        padded = {
            height: dimensions.height - padding * 2,
            width: dimensions.width - padding * 2
        };
        return padded;
    };

    dimensions = function(bounds, padding) {
        var padded;
        padded = pad(resolve_bounds(bounds), resolve_padding(padding));
        return padded;
    };

    wrap = {};

    // wrap text using foreignobject html
    wrap.foreignobject = function(text, dimensions, padding) {
        var content,
            parent,
            foreignobject,
            div,
            textAttrs;
        // extract our desired content from the single text element
        content = text.text();
        // preserve text attributes
        textAttrs = {
            x: text.attr('x'),
            y: text.attr('y'),
            fill: text.attr('fill'),
            'font-size': text.attr('font-size'),
            'font-family': text.attr('font-family'),
            'font-weight': text.attr('font-weight'),
            'text-anchor': text.attr('text-anchor'),
            'dominant-baseline': text.attr('dominant-baseline')
        };
        // remove the text node and replace with a foreign object
        parent = d3.select(text.node().parentNode);
        text.remove();
        foreignobject = parent.append('foreignObject');
        // add foreign object and set dimensions, position, etc
        foreignobject
            .attr('requiredFeatures', 'http://www.w3.org/TR/SVG11/feature#Extensibility')
            .attr('width', dimensions.width)
            .attr('height', dimensions.height)
            .attr('x', textAttrs.x || 0)
            .attr('y', textAttrs.y || 0);
        if (typeof padding === 'number') {
            foreignobject
                .attr('x', (+textAttrs.x || 0) + padding)
                .attr('y', (+textAttrs.y || 0) + padding);
        }
        // insert an HTML div
        div = foreignobject
            .append('xhtml:div');
        // set div to same dimensions as foreign object and preserve styles
        div
            .style('height', dimensions.height + 'px')
            .style('width', dimensions.width + 'px')
            .style('color', textAttrs.fill || 'black')
            .style('font-size', textAttrs['font-size'] || '16px')
            .style('font-family', textAttrs['font-family'] || 'inherit')
            .style('font-weight', textAttrs['font-weight'] || 'normal')
            .style('text-align', textAttrs['text-anchor'] === 'middle' ? 'center' : 
                                textAttrs['text-anchor'] === 'end' ? 'right' : 'left')
            .style('line-height', '1.2')
            .style('overflow', 'hidden')
            .style('word-wrap', 'break-word')
            // insert text content
            .html(content);
        return div;
    };

    // wrap text using tspans
    wrap.tspans = function(text, dimensions, padding) {
        var pieces,
            piece,
            line_width,
            tspan,
            previous_content,
            textAttrs,
            lines = [],
            currentLine = '',
            words,
            i,
            word,
            testLine,
            testWidth;
        
        // preserve text attributes
        textAttrs = {
            x: text.attr('x'),
            y: text.attr('y'),
            fill: text.attr('fill'),
            'font-size': text.attr('font-size'),
            'font-family': text.attr('font-family'),
            'font-weight': text.attr('font-weight'),
            'text-anchor': text.attr('text-anchor'),
            'dominant-baseline': text.attr('dominant-baseline')
        };
        
        // Get the text content and split into words
        words = text.text().split(/\s+/);
        text.text('');
        
        // Create a temporary tspan to measure text width
        var tempTspan = text.append('tspan').style('visibility', 'hidden');
        tempTspan
            .attr('fill', textAttrs.fill)
            .attr('font-size', textAttrs['font-size'])
            .attr('font-family', textAttrs['font-family'])
            .attr('font-weight', textAttrs['font-weight']);
        
        // Build lines that fit within the width constraint
        for (i = 0; i < words.length; i++) {
            word = words[i];
            testLine = currentLine + (currentLine ? ' ' : '') + word;
            tempTspan.text(testLine);
            testWidth = tempTspan.node().getComputedTextLength() || 0;
            
            if (testWidth > dimensions.width && currentLine) {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        }
        if (currentLine) {
            lines.push(currentLine);
        }
        
        // Remove temporary tspan
        tempTspan.remove();
        
        // Create tspans for each line
        for (i = 0; i < lines.length; i++) {
            tspan = text.append('tspan');
            tspan
                .attr('x', textAttrs.x || 0)
                .attr('y', i === 0 ? (textAttrs.y || 0) : null)
                .attr('dy', i === 0 ? '0' : '1.2em')
                .attr('fill', textAttrs.fill)
                .attr('font-size', textAttrs['font-size'])
                .attr('font-family', textAttrs['font-family'])
                .attr('font-weight', textAttrs['font-weight'])
                .attr('text-anchor', textAttrs['text-anchor'])
                .attr('dominant-baseline', textAttrs['dominant-baseline'])
                .text(lines[i]);
        }
        
        // Apply padding if specified
        if (typeof padding === 'number') {
            text
                .attr('y', (+textAttrs.y || 0) + padding)
                .attr('x', (+textAttrs.x || 0) + padding);
        }
    };

    // factory to generate text wrap functions
    textwrap = function() {
        // text wrap function instance
        var wrapper,
            bounds,
            padding;
        wrapper = function(targets) {
            targets.each(function() {
                d3.select(this).call(wrap[method], dimensions(bounds, padding), resolve_padding(padding));
            });
        };
        // get or set wrapping boundaries
        wrapper.bounds = function(new_bounds) {
            if (new_bounds) {
                if (verify_bounds(new_bounds)) {
                    bounds = new_bounds;
                    return wrapper;
                } else {
                    console.error('invalid text wrapping bounds');
                    return false;
                }
            } else {
                return bounds;
            }
        };
        // get or set padding applied on top of boundaries
        wrapper.padding = function(new_padding) {
            if (new_padding) {
                if (typeof new_padding === 'number' || typeof new_padding === 'function') {
                    padding = new_padding;
                    return wrapper;
                } else {
                    console.error('text wrap padding value must be either a number or a function');
                    return false;
                }
            } else {
                return padding;
            }
        };
        // get or set wrapping method
        wrapper.method = function(new_method) {
            if (new_method) {
                method = new_method;
                return wrapper;
            } else {
                return method;
            }
        };
        return wrapper;
    };

    // Add textwrap to D3
    d3.textwrap = textwrap;
})()
